use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::post};
use serde::{Deserialize, Serialize};

use crate::engine::llama::LlamaEngine;

const RAGAPI_DEFAULT_HOST: &str = "127.0.0.1";
const RAGAPI_DEFAULT_PORT: u16 = 9091;

// Structs for json requests
#[derive(Deserialize, Debug)]
struct EmbedRequest {
  model: String,
  input: String,
}

#[derive(Serialize, Debug)]
struct EmbedResponse {
  model: String,
  embeddings: Vec<f32>,
}

#[derive(Clone)]
pub struct RagApiState {
  llama: LlamaEngine,
}

pub struct RagApi<'r> {
  host: &'r str,
  port: u16,
  llama: Option<LlamaEngine>,
}

impl<'r> RagApi<'r> {
  pub fn new() -> Self {
    Self {
      host: RAGAPI_DEFAULT_HOST,
      port: RAGAPI_DEFAULT_PORT,
      llama: None,
    }
  }

  pub fn with_listen_host(mut self, address: &'r str) -> Self {
    self.host = address;
    self
  }

  pub fn with_listen_port(mut self, port: u16) -> Self {
    self.port = port;
    self
  }

  pub fn with_llama_engine(mut self, llama: LlamaEngine) -> Self {
    self.llama = Some(llama);
    self
  }

  pub async fn listen(self) {
    let st = RagApiState {
      llama: self.llama.unwrap().clone(),
    };

    let app: Router = Router::new()
      .route("/api/embeddings", post(api_embeddings))
      .route("/api/completion", post(api_completion))
      .with_state(st);

    let address = format!("{}:{}", self.host, self.port);

    let listener = tokio::net::TcpListener::bind(address.clone())
      .await
      .expect(format!("Could not bind to address {}", address).as_str());

    println!("[rsrag::api] listening on '{}'...", address);

    let _ = axum::serve(listener, app).await;
  }
}

async fn api_embeddings(
  State(st): State<RagApiState>,
  Json(payload): Json<EmbedRequest>,
) -> impl IntoResponse {
  println!("[api_embeddings] payload is {:?}", payload);

  (StatusCode::OK, "")
}

async fn api_completion(
  State(st): State<RagApiState>,
  Json(payload): Json<EmbedRequest>,
) -> impl IntoResponse {
  println!("[api_completion] payload is {:?}", payload);

  (StatusCode::OK, "")
}
