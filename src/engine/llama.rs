use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Serialize, Deserialize)]
struct LlamaEmbedRequest {
  input: String,
}

// Constants
const LLAMA_DEFAULT_SERVER: &str = "http://127.0.0.1:8080";

#[derive(Clone)]
pub struct LlamaEngine {
  embed_server: String,
  llama_server: String,
  rerank_server: String,
  chat_model: Option<String>,
  embed_model: Option<String>,
  rerank_model: Option<String>,
}

impl<'l> LlamaEngine {
  pub fn new() -> Self {
    Self {
      embed_server: LLAMA_DEFAULT_SERVER.into(),
      llama_server: LLAMA_DEFAULT_SERVER.into(),
      rerank_server: LLAMA_DEFAULT_SERVER.into(),
      rerank_model: None,
      chat_model: None,
      embed_model: None,
    }
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
    self.chat_model = Some(s.into());
    self
  }

  pub fn print(self) {
    println!("--- LlamaEngine ---");
    println!("embed_server ..... {}", self.embed_server);
    println!("llama_server ..... {}", self.llama_server);
    println!("rerank_server .... {}", self.rerank_server);
    println!(
      "chat_model ....... {}",
      self.chat_model.unwrap_or("undefined".into())
    );
    println!(
      "embed_model ...... {}",
      self.embed_model.unwrap_or("undefined".into())
    );
    println!(
      "rerank_model ..... {}",
      self.rerank_model.unwrap_or("undefined".into())
    );
  }

  // Endpoints implementation
  pub async fn get_embeddings(
    self,
    input: &str,
  ) -> Result<Option<Vec<f32>>, Box<dyn std::error::Error>> {
    let url = format!("{}/v1/embeddings", self.embed_server);
    let req = LlamaEmbedRequest {
      input: input.into(),
    };

    println!("making request to {}", url);

    let client = Client::new()
      .post(url)
      .header("Content-Type", "application/json")
      .body("{\"input\": \"busco sexo\"}")
      .timeout(Duration::from_secs(5))
      .send()
      .await?;

    let body = client.text().await?;

    println!("response body: {}", body);

    Ok(None)
  }
}
