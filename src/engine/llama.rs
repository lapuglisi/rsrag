use axum::{body::Body, response::IntoResponse};
use futures_util::Stream;
use http::StatusCode;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

//
// Llama completion
//

const LLAMA_DEFAULT_TEMPERATURE: f32 = 0.8;
const LLAMA_DEFAULT_NPREDICT: i32 = -1;
const LLAMA_DEFAULT_TOP_K: u32 = 40;
const LLAMA_DEFAULT_TOP_P: f32 = 0.9;
const LLAMA_DEFAULT_STREAM: bool = false;

#[derive(Serialize, Debug)]
#[serde(default)]
pub struct LlamaCompletionRequest {
  pub model: String,
  pub messages: Vec<LlamaCompletionMessage>,
  pub temperature: f32,
  pub n_predict: i32,
  pub top_k: u32,
  pub top_p: f32,
  pub stream: bool,
}

impl Default for LlamaCompletionRequest {
  fn default() -> Self {
    Self {
      model: String::new(),
      messages: Vec::new(),
      temperature: 0.8,
      n_predict: -1,
      top_k: 40,
      top_p: 0.9,
      stream: false,
    }
  }
}

impl LlamaCompletionRequest {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn with_model(mut self, m: &str) -> Self {
    self.model = m.into();
    self
  }

  pub fn add_message(mut self, msg: LlamaCompletionMessage) -> Self {
    self.messages.push(msg);
    self
  }

  pub fn append_messages(mut self, mut msgs: Vec<LlamaCompletionMessage>) -> Self {
    self.messages.append(&mut msgs);
    self
  }

  pub fn with_temperature(mut self, t: Option<f32>) -> Self {
    self.temperature = t.unwrap_or(LLAMA_DEFAULT_TEMPERATURE);
    self
  }

  pub fn with_n_predict(mut self, p: Option<i32>) -> Self {
    self.n_predict = p.unwrap_or(LLAMA_DEFAULT_NPREDICT);
    self
  }

  pub fn with_top_k(mut self, k: Option<u32>) -> Self {
    self.top_k = k.unwrap_or(LLAMA_DEFAULT_TOP_K);
    self
  }

  pub fn with_top_p(mut self, p: Option<f32>) -> Self {
    self.top_p = p.unwrap_or(LLAMA_DEFAULT_TOP_P);
    self
  }

  pub fn stream(mut self, s: Option<bool>) -> Self {
    self.stream = s.unwrap_or(LLAMA_DEFAULT_STREAM);
    self
  }
}

#[derive(Serialize, Debug)]
pub struct LlamaCompletionMessage {
  pub role: String,
  pub content: String,
}

impl Default for LlamaCompletionMessage {
  fn default() -> Self {
    Self {
      role: String::new(),
      content: String::new(),
    }
  }
}

impl LlamaCompletionMessage {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn with_role(mut self, role: &str) -> Self {
    self.role = role.to_string();
    self
  }

  pub fn with_content(mut self, content: &str) -> Self {
    self.content = content.to_string();
    self
  }
}

//
//
//

#[derive(Serialize, Deserialize)]
pub struct LlamaEmbedRequest {
  pub input: String,
}

//
// Llama Embedding Response
//
#[derive(Deserialize, Debug)]
struct LlamaEmbedResponseData {
  index: u32,
  object: String,
  embedding: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct LlamaEmbedResponseUsage {
  prompt_tokens: u32,
  total_tokens: u32,
}

#[derive(Deserialize, Debug)]
struct LlamaEmbedResponse {
  model: String,
  object: String,
  usage: LlamaEmbedResponseUsage,
  data: Vec<LlamaEmbedResponseData>,
}
////////////////////////////

#[derive(Clone, Default)]
pub struct LlamaEngine {
  pub embed_server: String,
  pub llama_server: String,
  pub rerank_server: String,
  pub chat_model: String,
  pub embed_model: String,
  pub rerank_model: String,
}

impl<'l> LlamaEngine {
  pub fn new() -> Self {
    LlamaEngine::default()
  }

  pub fn with_llama_server(mut self, s: &str) -> Self {
    self.llama_server = s.into();
    self
  }

  pub fn with_embed_server(mut self, s: &str) -> Self {
    self.embed_server = s.into();
    self
  }

  pub fn with_rerank_server(mut self, s: &str) -> Self {
    self.rerank_server = s.into();
    self
  }

  pub fn with_chat_model(mut self, s: &str) -> Self {
    self.chat_model = s.into();
    self
  }

  pub fn with_embed_model(mut self, s: &str) -> Self {
    self.embed_model = s.into();
    self
  }

  pub fn with_rerank_model(mut self, s: &str) -> Self {
    self.rerank_model = s.into();
    self
  }

  pub fn print(self) {
    println!("--- LlamaEngine ---");
    println!("embed_server ..... {}", self.embed_server);
    println!("llama_server ..... {}", self.llama_server);
    println!("rerank_server .... {}", self.rerank_server);
    println!("chat_model ....... {}", self.chat_model);
    println!("embed_model ...... {}", self.embed_model);
    println!("rerank_model ..... {}", self.rerank_model);
  }

  // Endpoints implementation
  pub async fn get_embeddings(
    self,
    input: &str,
  ) -> Result<crate::api::EmbedResponse, Box<dyn std::error::Error>> {
    let url = format!("{}/v1/embeddings", self.embed_server);
    let er = LlamaEmbedRequest {
      input: input.into(),
    };

    let json = serde_json::to_string(&er)?;

    log::info!("sending request to {}", url);
    log::info!("json is {}", json);

    let client = Client::new()
      .post(url)
      .header("Content-Type", "application/json")
      .body(json)
      .timeout(Duration::from_secs(60))
      .send()
      .await?;

    let body = client.text().await?;
    let erp: LlamaEmbedResponse = serde_json::from_str(&body)?;

    let embed = crate::api::EmbedResponse {
      model: erp.model,
      embeddings: erp.data[0].embedding.clone(),
    };

    log::debug!("returning response: {:?}", embed);

    Ok(embed)
  }

  pub async fn get_completion(self, request: LlamaCompletionRequest) -> impl IntoResponse {
    let url = format!("{}/v1/chat/completions", self.llama_server);

    log::info!("send {:?} to {}", request, url);

    let json = serde_json::to_string(&request).unwrap_or(String::new());

    let client = Client::new().post(url).body(json).send().await;

    let response = match client {
      Ok(r) => Body::from_stream(r.bytes_stream()).into_response(),
      Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("error: {}", e)).into_response(),
    };

    response
  }
}
