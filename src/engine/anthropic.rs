#![allow(unused)]
use axum::{body::Body, response::IntoResponse};
use futures_util::{StreamExt, TryFutureExt};
use http::StatusCode;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::engine::AICompletionMessage;

//
// Anthropic completion
//

pub const ANTHROPIC_DEFAULT_VERSION: &str = "2026-10-21";

pub const ANTHROPIC_DEFAULT_MAX_TOKENS: i32 = 4096;
pub const ANTHROPIC_DEFAULT_STREAM: bool = false;

#[derive(Serialize, Debug)]
#[serde(default)]
pub struct AnthropicCompletionRequest {
  pub model: String,
  pub messages: Vec<AICompletionMessage>,
  pub max_tokens: i32,
  pub stream: bool,
}

impl Default for AnthropicCompletionRequest {
  fn default() -> Self {
    Self {
      model: String::new(),
      messages: Vec::new(),
      max_tokens: ANTHROPIC_DEFAULT_MAX_TOKENS,
      stream: ANTHROPIC_DEFAULT_STREAM,
    }
  }
}

impl AnthropicCompletionRequest {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn with_model(mut self, m: &str) -> Self {
    self.model = m.into();
    self
  }

  #[allow(unused)]
  pub fn add_message(mut self, msg: AICompletionMessage) -> Self {
    self.messages.push(msg);
    self
  }

  pub fn append_messages(mut self, mut msgs: Vec<AICompletionMessage>) -> Self {
    self.messages.append(&mut msgs);
    self
  }

  pub fn with_max_tokens(mut self, m: Option<i32>) -> Self {
    self.max_tokens = m.unwrap_or(ANTHROPIC_DEFAULT_MAX_TOKENS);
    self
  }

  pub fn stream(mut self, s: Option<bool>) -> Self {
    self.stream = s.unwrap_or(ANTHROPIC_DEFAULT_STREAM);
    self
  }
}

#[derive(Serialize, Debug)]
pub struct AnthropicCompletionMessage {
  pub role: String,
  pub content: String,
}

impl Default for AnthropicCompletionMessage {
  fn default() -> Self {
    Self {
      role: String::new(),
      content: String::new(),
    }
  }
}

impl AnthropicCompletionMessage {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn new_with_role(role: &str) -> Self {
    Self {
      role: role.to_string(),
      content: String::new(),
    }
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

#[derive(Clone, Default, Debug)]
pub struct AnthropicEngine {
  pub endpoint: String,
  pub api_key: String,
  pub version: String,
}

impl AnthropicEngine {
  pub fn new() -> Self {
    Self {
      endpoint: String::new(),
      api_key: String::new(),
      version: String::from(ANTHROPIC_DEFAULT_VERSION),
    }
  }

  pub fn with_endpoint(mut self, s: &str) -> Self {
    self.endpoint = s.into();
    self
  }

  pub fn with_api_key(mut self, s: &str) -> Self {
    self.api_key = s.into();
    self
  }

  pub fn with_version(mut self, v: &str) -> Self {
    self.version = v.into();
    self
  }

  pub fn auth_header(&self) -> String {
    format!("Bearer {}", self.api_key)
  }

  #[allow(unused)]
  pub fn print(self) {
    println!("--- AnthropicEngine ---");
    println!("endpoint ..... {}", self.endpoint);
    println!("api_key ...... {}", self.api_key);
  }

  pub async fn get_completion(&self, request: AnthropicCompletionRequest) -> impl IntoResponse {
    let url = format!("{}/messages", self.endpoint);

    log::debug!("send {:?} to {}", request, url);

    let json = match serde_json::to_string(&request) {
      Ok(j) => j,
      Err(e) => {
        let error = format!("could not serialize json: {}", e);
        log::warn!("{}", error);
        return Body::from(error).into_response();
      }
    };

    log::info!("Anthropic: sending JSON payload: {}", json);

    let client = Client::new()
      .post(url)
      .header("Content-Type", "application/json")
      .header("Authorization", self.auth_header())
      .header("x-api-key", &self.api_key)
      .header("anthropic-version", &self.version)
      .body(json);

    log::info!("Anthropic client: {:?}", client);

    match client.send().await {
      Ok(r) => Body::from_stream(r.bytes_stream()).into_response(),
      Err(e) => {
        log::warn!("error getting completion: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, format!("error: {}", e)).into_response()
      }
    }
  }

  pub async fn get_models(&self) -> impl IntoResponse {
    let url = format!("{}/models", self.endpoint);

    match Client::new()
      .get(url)
      .header("Authorization", self.auth_header())
      .send()
      .await
    {
      Ok(r) => Body::from_stream(r.bytes_stream()).into_response(),
      Err(e) => {
        log::error!("error while retrieving models: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, "error").into_response()
      }
    }
  }
}
