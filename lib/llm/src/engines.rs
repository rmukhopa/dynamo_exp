// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::env;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Duration;

use async_stream::stream;
use async_trait::async_trait;
use async_openai::types::Prompt;
use futures::StreamExt;
use std::pin::Pin;

use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn, AsyncEngineContext};
use dynamo_runtime::protocols::annotated::Annotated;

use crate::backend::ExecutionContext;
use crate::preprocessor::BackendInput;
use crate::protocols::common::llm_backend::LLMEngineOutput;
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse
};
use crate::protocols::openai::completions::{CompletionRequest, CompletionResponse, DeltaGenerator};
use crate::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine;
use crate::types::openai::completions::OpenAICompletionsStreamingEngine;
use dynamo_runtime::engine::AsyncEngineStream;  
use dynamo_runtime::{pipeline::Context};
use dynamo_runtime::pipeline::ServerStreamingEngine; 
//
// The engines are each in their own crate under `lib/engines`
//

#[derive(Debug, Clone)]
pub struct MultiNodeConfig {
    /// How many nodes / hosts we are using
    pub num_nodes: u32,
    /// Unique consecutive integer to identify this node
    pub node_rank: u32,
    /// host:port of head / control node
    pub leader_addr: String,
}

impl Default for MultiNodeConfig {
    fn default() -> Self {
        MultiNodeConfig {
            num_nodes: 1,
            node_rank: 0,
            leader_addr: "".to_string(),
        }
    }
}

//
// Example echo engines
//

/// How long to sleep between echoed tokens.
/// Default is 10ms which gives us 100 tok/s.
/// Can be configured via the DYN_TOKEN_ECHO_DELAY_MS environment variable.
pub static TOKEN_ECHO_DELAY: LazyLock<Duration> = LazyLock::new(|| {
    const DEFAULT_DELAY_MS: u64 = 10;

    let delay_ms = env::var("DYN_TOKEN_ECHO_DELAY_MS")
        .ok()
        .and_then(|val| val.parse::<u64>().ok())
        .unwrap_or(DEFAULT_DELAY_MS);

    Duration::from_millis(delay_ms)
});

/// Engine that accepts pre-processed requests and echos the tokens back as the response
/// The response will include the full prompt template.
/// Useful for testing pre-processing.
struct EchoEngineCore {}
pub fn make_engine_core() -> ExecutionContext {
    Arc::new(EchoEngineCore {})
}

#[async_trait]
impl AsyncEngine<SingleIn<BackendInput>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for EchoEngineCore
{
    async fn generate(
        &self,
        incoming_request: SingleIn<BackendInput>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (request, context) = incoming_request.into_parts();
        let ctx = context.context();

        let output = stream! {
            for tok in request.token_ids {
                tokio::time::sleep(*TOKEN_ECHO_DELAY).await;
                yield delta_core(tok);
            }
            yield Annotated::from_data(LLMEngineOutput::stop());
        };
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

fn delta_core(tok: u32) -> Annotated<LLMEngineOutput> {
    let delta = LLMEngineOutput {
        token_ids: vec![tok],
        tokens: None,
        text: None,
        cum_log_probs: None,
        log_probs: None,
        finish_reason: None,
    };
    Annotated::from_data(delta)
}

/// Engine that accepts un-preprocessed requests and echos the prompt back as the response
/// Useful for testing ingress such as service-http.
struct EchoEngineFull {}
pub fn make_engine_full() -> Arc<dyn StreamingEngine> {
    Arc::new(UnifiedEngine::new(EchoEngineFull {}))
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for EchoEngineFull
{
    async fn generate(
        &self,
        incoming_request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (request, context) = incoming_request.transfer(());
        let deltas = request.response_generator();
        let ctx = context.context();
        let req = request.inner.messages.into_iter().next_back().unwrap();

        let prompt = match req {
            async_openai::types::ChatCompletionRequestMessage::User(user_msg) => {
                match user_msg.content {
                    async_openai::types::ChatCompletionRequestUserMessageContent::Text(prompt) => {
                        prompt
                    }
                    _ => anyhow::bail!("Invalid request content field, expected Content::Text"),
                }
            }
            _ => anyhow::bail!("Invalid request type, expected User message"),
        };

        let output = stream! {
            let mut id = 1;
            for c in prompt.chars() {
                // we are returning characters not tokens, so there will be some postprocessing overhead
                tokio::time::sleep(*TOKEN_ECHO_DELAY).await;
                let inner = deltas.create_choice(0, Some(c.to_string()), None, None);
                let response = NvCreateChatCompletionStreamResponse {
                    inner,
                };
                yield Annotated{ id: Some(id.to_string()), data: Some(response), event: None, comment: None };
                id += 1;
            }

            let inner = deltas.create_choice(0, None, Some(async_openai::types::FinishReason::Stop), None);
            let response = NvCreateChatCompletionStreamResponse {
                inner,
            };
            yield Annotated { id: Some(id.to_string()), data: Some(response), event: None, comment: None };
        };

        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}


struct InternalRequest {
    text: String,
    response_generator: DeltaGenerator, 
    context: Context<()>,
}

struct InternalResponse {
    text: String,
    is_final: bool,
    response_generator: DeltaGenerator,
}

pub type UnifiedStreamingEngine = ServerStreamingEngine<
    InternalRequest,
    Annotated<InternalResponse>,
>;


impl From<SingleIn<CompletionRequest>> for InternalRequest {
    fn from(incoming_request: SingleIn<CompletionRequest>) -> Self {
        let (request, context) = incoming_request.transfer(());
        let chars_string = match request.inner.prompt {
            Prompt::String(ref s) => s,
            Prompt::StringArray(ref arr) => &arr.join(""),
            Prompt::IntegerArray(_) => panic!("Integer array prompts not supported"),
            Prompt::ArrayOfIntegerArray(_) => panic!("Array of integer array prompts not supported"),
        };
        InternalRequest {
            text: chars_string.to_string(),
            response_generator: request.response_generator(),
            context: context,
        }
    }
}


impl From<InternalResponse> for CompletionResponse {
    fn from(response: InternalResponse) -> CompletionResponse {
        if response.is_final {
            response.response_generator.create_choice(0, Some(response.text), None)
        } else {
            response.response_generator.create_choice(0, None, Some("stop".to_string()))
        }
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<InternalRequest>, ManyOut<Annotated<InternalResponse>>, Error>
    for EchoEngineFull
{
    async fn generate(
        &self,
        incoming_request: SingleIn<InternalRequest>,
    ) -> Result<ManyOut<Annotated<InternalResponse>>, Error> {
        let (request, context) = incoming_request.transfer(());
        let chars_string = request.text;
        let deltas = request.response_generator;
        let ctx = context.context();

        let output = stream! {
            let mut id = 1;
            for c in chars_string.chars() {
                tokio::time::sleep(*TOKEN_ECHO_DELAY).await;
                let response = InternalResponse{text: c.to_string(), is_final: false, response_generator: deltas.clone() };
                yield Annotated{ id: Some(id.to_string()), data: Some(response), event: None, comment: None };
                id += 1;
            }

            let response = InternalResponse{text: "".to_string(), is_final: true, response_generator: deltas };
            yield Annotated { id: Some(id.to_string()), data: Some(response), event: None, comment: None };
        };

        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

// Add this struct definition before the StreamingEngine trait
struct UnifiedEngine<E> {
    inner: E,
}

// Constructor for convenience
impl<E> UnifiedEngine<E> {
    pub fn new(inner: E) -> Self {
        UnifiedEngine { inner }
    }
}

// Your existing trait and implementation
#[async_trait]
pub trait StreamingEngine: Send + Sync {
    async fn handle_completion(&self, req: SingleIn<CompletionRequest>) 
        -> Result<ManyOut<Annotated<CompletionResponse>>, Error>;
    
    async fn handle_chat(&self, req: SingleIn<NvCreateChatCompletionRequest>)
        -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error>;
}

#[async_trait]
impl<E> StreamingEngine for UnifiedEngine<E> 
where 
    E: AsyncEngine<SingleIn<InternalRequest>, ManyOut<Annotated<InternalResponse>>, Error> + Send + Sync,
{
    async fn handle_completion(&self, req: SingleIn<CompletionRequest>) 
        -> Result<ManyOut<Annotated<CompletionResponse>>, Error>
    {
        let internal_request = InternalRequest::from(req);
        let ctx = internal_request.context.context();

        let internal_stream = self.inner.generate(SingleIn::new(internal_request)).await?;
        let output = internal_stream.map(|annotated| {
            Annotated {
                id: annotated.id,
                data: annotated.data.map(CompletionResponse::from),
                event: annotated.event,
                comment: annotated.comment,
            }
        });

        Ok(ResponseStream::new(Box::pin(output), ctx))
    }

    async fn handle_chat(&self, req: SingleIn<NvCreateChatCompletionRequest>)
        -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error>
    {
        let (request, context) = req.transfer(());
        let deltas = request.response_generator();
        let ctx = context.context();
        let req = request.inner.messages.into_iter().next_back().unwrap();

        let prompt = match req {
            async_openai::types::ChatCompletionRequestMessage::User(user_msg) => {
                match user_msg.content {
                    async_openai::types::ChatCompletionRequestUserMessageContent::Text(prompt) => {
                        prompt
                    }
                    _ => anyhow::bail!("Invalid request content field, expected Content::Text"),
                }
            }
            _ => anyhow::bail!("Invalid request type, expected User message"),
        };

        let output = stream! {
            let mut id = 1;
            for c in prompt.chars() {
                // we are returning characters not tokens, so there will be some postprocessing overhead
                tokio::time::sleep(*TOKEN_ECHO_DELAY).await;
                let inner = deltas.create_choice(0, Some(c.to_string()), None, None);
                let response = NvCreateChatCompletionStreamResponse {
                    inner,
                };
                yield Annotated{ id: Some(id.to_string()), data: Some(response), event: None, comment: None };
                id += 1;
            }

            let inner = deltas.create_choice(0, None, Some(async_openai::types::FinishReason::Stop), None);
            let response = NvCreateChatCompletionStreamResponse {
                inner,
            };
            yield Annotated { id: Some(id.to_string()), data: Some(response), event: None, comment: None };
        };

        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

// Keep only one wrapper type
pub struct StreamingEngineAdapter(Arc<dyn StreamingEngine>);

impl StreamingEngineAdapter {
    pub fn new(engine: Arc<dyn StreamingEngine>) -> Self {
        StreamingEngineAdapter(engine)
    }
}

// Implement AsyncEngine for our wrapper
#[async_trait]
impl AsyncEngine<SingleIn<CompletionRequest>, ManyOut<Annotated<CompletionResponse>>, Error> 
    for StreamingEngineAdapter 
{
    async fn generate(&self, req: SingleIn<CompletionRequest>) 
        -> Result<ManyOut<Annotated<CompletionResponse>>, Error> 
    {
        self.0.handle_completion(req).await
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<NvCreateChatCompletionRequest>, ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> 
    for StreamingEngineAdapter 
{
    async fn generate(&self, req: SingleIn<NvCreateChatCompletionRequest>) 
        -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> 
    {
        self.0.handle_chat(req).await
    }
}
