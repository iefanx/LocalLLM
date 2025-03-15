/**
 * Web worker for handling WebLLM engine operations in a separate thread
 * This prevents UI freezing during intensive LLM operations
 */
import { WebWorkerMLCEngineHandler } from "@mlc-ai/web-llm";

// Initialize the engine handler
const handler = new WebWorkerMLCEngineHandler();

// Handle incoming messages from the main thread
self.onmessage = (msg: MessageEvent) => {
  try {
    handler.onmessage(msg);
  } catch (error) {
    // Post error back to main thread
    self.postMessage({
      type: "error",
      error: error instanceof Error ? error.message : String(error),
      id: (msg.data && typeof msg.data === 'object' && 'id' in msg.data) ? msg.data.id : undefined
    });
    
    console.error("Worker error:", error);
  }
};

// Handle unhandled errors in the worker
self.addEventListener('error', (event) => {
  console.error('Worker global error:', event.error || event.message);
});

// Log worker initialization
console.log("WebLLM worker initialized");
