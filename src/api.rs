#![allow(dead_code)]
use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::post};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::engine::{
  llama::{
    LLAMA_DEFAULT_NPREDICT, LLAMA_DEFAULT_STREAM, LLAMA_DEFAULT_TEMPERATURE, LLAMA_DEFAULT_TOP_K,
    LLAMA_DEFAULT_TOP_P, LLAMA_RERANK_DEFAULT_THRESHOLD, LlamaCompletionMessage,
    LlamaCompletionRequest, LlamaEngine, RerankRequest,
  },
  qdrant::{QDRANT_QUERY_DEFAULT_THRESHOLD, QdrantEngine, QdrantQuery},
};

const RAGAPI_DEFAULT_HOST: &str = "127.0.0.1";
const RAGAPI_DEFAULT_PORT: u16 = 9091;
const RAGAPI_LLM_SYSTEM_PROMPT: &str = "You are a helpful assistant and expert in answering prompts in a RAG environment.\nAnswer the user prompt using only the provided context.\nIf you do not know the answer, simply state: I don't know.\nMAKE SURE to answer the user's question in the original language.";

// Structs for json requests
#[derive(Deserialize, Debug)]
struct EmbedRequest {
  model: String,
  input: String,
}

#[derive(Serialize, Debug)]
pub struct EmbedResponse {
  pub model: String,
  pub embeddings: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
pub struct CompletionRequest {
  pub model: String,
  pub prompt: String,
  pub temperature: Option<f32>,
  pub stream: Option<bool>,
  pub n_predict: Option<i32>,
  pub top_k: Option<u32>,
  pub top_p: Option<f32>,
  pub threshold: Option<f32>,
  pub rerank: Option<CompletionRequestRerank>,
}

impl Default for CompletionRequest {
  fn default() -> Self {
    Self {
      model: String::new(),
      prompt: String::new(),
      temperature: Some(LLAMA_DEFAULT_TEMPERATURE),
      stream: Some(LLAMA_DEFAULT_STREAM),
      n_predict: Some(LLAMA_DEFAULT_NPREDICT),
      top_k: Some(LLAMA_DEFAULT_TOP_K),
      top_p: Some(LLAMA_DEFAULT_TOP_P),
      threshold: Some(QDRANT_QUERY_DEFAULT_THRESHOLD),
      rerank: None,
    }
  }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(default)]
pub struct CompletionRequestRerank {
  model: String,
  threshold: Option<f32>,
}

impl Default for CompletionRequestRerank {
  fn default() -> Self {
    Self {
      model: String::new(),
      threshold: Some(LLAMA_RERANK_DEFAULT_THRESHOLD),
    }
  }
}

#[derive(Clone)]
pub struct RagApiState {
  llama: Arc<LlamaEngine>,
  qdrant: Arc<QdrantEngine>,
}

pub struct RagApi<'r> {
  host: &'r str,
  port: u16,
  llama: Option<LlamaEngine>,
  qdrant: Option<QdrantEngine>,
}

impl<'r> RagApi<'r> {
  pub fn new() -> Self {
    Self {
      host: RAGAPI_DEFAULT_HOST,
      port: RAGAPI_DEFAULT_PORT,
      llama: None,
      qdrant: None,
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

  pub fn with_qdrant_engine(mut self, q: QdrantEngine) -> Self {
    self.qdrant = Some(q);
    self
  }

  pub async fn listen(self) -> Result<(), Box<dyn std::error::Error>> {
    if self.llama.is_none() {
      Err("[rsrag::api::listen] no llama engine configured")?
    }

    if self.qdrant.is_none() {
      Err("[rsrag::api::listen] no qdrant engine configured.")?
    }

    let st = RagApiState {
      llama: Arc::new(self.llama.unwrap()),
      qdrant: Arc::new(self.qdrant.unwrap()),
    };

    let app: Router = Router::new()
      .route("/api/embeddings", post(api_embeddings))
      .route("/api/completion", post(api_completion))
      .with_state(st);

    let address = format!("{}:{}", self.host, self.port);

    let listener = tokio::net::TcpListener::bind(&address)
      .await
      .map_err(|err| {
        format!(
          "[rsrag::api]: Could not bind to address {}: {}",
          address,
          err.to_string()
        )
      })?;

    println!("[rsrag::api] listening on '{}'...", address);
    log::info!("[rsrag::api] listening on '{}'...", address);

    let _ = axum::serve(listener, app).await;

    Ok(())
  }
}

fn get_qdrant_colletion(model: &str) -> String {
  model
    .chars()
    .map(|c| if c.is_ascii_punctuation() { '-' } else { c })
    .collect()
}

#[axum::debug_handler]
async fn api_embeddings(
  State(st): State<RagApiState>,
  Json(payload): Json<EmbedRequest>,
) -> impl IntoResponse {
  log::debug!("[api_embeddings] payload is {:?}", payload);

  let resp = match st.llama.get_embeddings(&payload.input).await {
    Ok(v) => (
      StatusCode::OK,
      serde_json::to_string(&v).unwrap_or("{}".to_string()),
    ),
    Err(e) => (
      StatusCode::INTERNAL_SERVER_ERROR,
      format!("embeddings error: {}", e),
    ),
  };

  resp.into_response()
}

#[axum::debug_handler]
async fn api_completion(
  State(st): State<RagApiState>,
  Json(payload): Json<CompletionRequest>,
) -> impl IntoResponse {
  log::info!("[api_completion] payload is {:?}", payload);

  // get embeddings for prompt
  let llama = st.llama;
  let qdrant = st.qdrant;
  let mut messages: Vec<LlamaCompletionMessage> = Vec::new();

  if qdrant.client.is_none() {
    return (
      StatusCode::INTERNAL_SERVER_ERROR,
      "qdrant client unitialized",
    )
      .into_response();
  }

  let er: EmbedResponse = match llama.get_embeddings(&payload.prompt).await {
    Ok(v) => v,
    Err(e) => {
      log::warn!("could not retrieve embeddings: {}", e);
      EmbedResponse {
        model: "none".to_string(),
        embeddings: Vec::new(),
      }
    }
  };

  // Do qdrant similarity search
  let collection: String = get_qdrant_colletion(&er.model);
  log::info!("using qdrant collection {}", &collection);

  match QdrantQuery::new(&qdrant)
    .with_collection(collection)
    .with_embeddings(er.embeddings)
    .with_threshold(payload.threshold.unwrap())
    .run()
    .await
  {
    Ok(res) => {
      log::debug!("qdrant::query_points yielded {:?}", res);

      let docs: Vec<String> = res
        .iter()
        .filter_map(|item| {
          if let Some(v) = item.payload["source"].as_str() {
            Some(v.to_string())
          } else {
            None
          }
        })
        .collect();

      let context = if payload.rerank.is_some() {
        let rerank = payload.rerank.unwrap();

        log::debug!("reranking vectordb results...");

        let rr = RerankRequest::new()
          .with_query(&payload.prompt)
          .with_rerank_model(rerank.model)
          .with_threshold(rerank.threshold.unwrap())
          .add_documents(&docs);

        log::debug!("rerank request: {:?}", rr);

        match llama.rerank(rr).await {
          Ok(r) => r,
          Err(e) => {
            log::warn!("could not rerank results: {}", e);
            // Use initial documents as context
            docs
          }
        }
      } else {
        log::warn!("not reranking qdrant result: no rerank config provided.");
        docs
      };

      messages.push(
        LlamaCompletionMessage::new()
          .with_role("system")
          .with_content(RAGAPI_LLM_SYSTEM_PROMPT),
      );

      messages.push(
        LlamaCompletionMessage::new()
          .with_role("user")
          .with_content("Hi, LLM."),
      );
      messages.push(
        LlamaCompletionMessage::new()
          .with_role("assistant")
          .with_content("Hi, user."),
      );

      messages.push(
        LlamaCompletionMessage::new()
          .with_role("user")
          .with_content(
            "Provided some context, answer th user's query with as much information as you can.",
          ),
      );

      messages.push(
        LlamaCompletionMessage::new()
          .with_role("assistant")
          .with_content("Sure! What is the context?"),
      );

      messages.push(
        LlamaCompletionMessage::new()
          .with_role("user")
          .with_content(context.join("\n").as_str()),
      );

      messages.push(
        LlamaCompletionMessage::new()
          .with_role("assistant")
          .with_content("Perfetct. And what do you want to know?"),
      );

      messages.push(
        LlamaCompletionMessage::new()
          .with_role("user")
          .with_content(&payload.prompt),
      );
    }
    Err(e) => {
      log::warn!("similarity search: {}", e);
      messages.push(
        LlamaCompletionMessage::new()
          .with_role("user")
          .with_content(&payload.prompt),
      );
    }
  }

  // Prepare a completion request
  let lcr = LlamaCompletionRequest::new()
    .with_model(&payload.model)
    .with_temperature(payload.temperature)
    .with_top_k(payload.top_k)
    .with_top_p(payload.top_p)
    .with_n_predict(payload.n_predict)
    .stream(payload.stream)
    .append_messages(messages);

  log::debug!("request llama with {:?}", lcr);

  llama.get_completion(lcr).await.into_response()
}
