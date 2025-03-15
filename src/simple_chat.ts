import appConfig from "./app-config";
import * as webllm from "@mlc-ai/web-llm";
import { refreshIcons } from "./icons"; // Import the icon refresh function
import { addSession, getSessions, loadSession as loadSessionFromDB, updateSession } from "./chat-session-db";

// Constants
const SELECTED_MODEL_KEY = "web-llm-selected-model";
const ANDROID_MAX_STORAGE_BUFFER_SIZE = 1 << 27; // 128MB
const MOBILE_GPU_VENDORS = new Set<string>(["qualcomm", "arm"]);

/**
 * Get DOM element by ID with null check
 */
function getElementAndCheck<T extends HTMLElement = HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element == null) {
    throw Error(`Cannot find element with ID: ${id}`);
  }
  return element as T;
}

/**
 * Message type enum for chat UI
 */
type MessageKind = "left" | "right" | "init" | "error";

/**
 * ChatUI Class - manages the chat interface and interactions with the WebLLM engine
 */
class ChatUI {
  private readonly uiChat: HTMLElement;
  private readonly uiChatInput: HTMLInputElement;
  private readonly uiChatInfoLabel: HTMLLabelElement;
  private readonly engine: webllm.MLCEngineInterface | webllm.WebWorkerMLCEngine;
  private readonly config: webllm.AppConfig = appConfig;
  
  private selectedModel: string = "";
  private chatLoaded = false;
  private requestInProgress = false;
  private chatHistory: webllm.ChatCompletionMessageParam[] = [];
  
  private readonly uiSidebarModelSelect: HTMLSelectElement;
  private currentSessionId: number | null = null;
  
  // We use a request chain to ensure that all requests to chat are sequentialized
  private chatRequestChain: Promise<void> = Promise.resolve();

  /**
   * Private constructor - use CreateAsync factory method instead
   */
  private constructor(engine: webllm.MLCEngineInterface | webllm.WebWorkerMLCEngine) {
    this.engine = engine;
    this.uiChat = getElementAndCheck("chatui-chat");
    this.uiChatInput = getElementAndCheck<HTMLInputElement>("chatui-input");
    this.uiChatInfoLabel = getElementAndCheck<HTMLLabelElement>("chatui-info-label");
    
    // Get sidebar model select if it exists
    const sidebarModelSelect = document.getElementById("sidebar-model-select");
    if (sidebarModelSelect) {
      this.uiSidebarModelSelect = sidebarModelSelect as HTMLSelectElement;
    }
  }

  /**
   * Save the selected model to local storage
   */
  private saveSelectedModel(modelId: string): void {
    try {
      localStorage.setItem(SELECTED_MODEL_KEY, modelId);
    } catch (e) {
      console.warn("Failed to save model preference to localStorage:", e);
    }
  }

  /**
   * Get the saved model ID from local storage
   */
  private static getSavedModelId(): string | null {
    try {
      return localStorage.getItem(SELECTED_MODEL_KEY);
    } catch (e) {
      console.warn("Failed to read model preference from localStorage:", e);
      return null;
    }
  }

  /**
   * An asynchronous factory constructor since initialization requires awaiting async operations
   */
  public static async CreateAsync(engine: webllm.MLCEngineInterface): Promise<ChatUI | undefined> {
    const chatUI = new ChatUI(engine);
    
    // Register event handlers
    getElementAndCheck("chatui-reset-btn").onclick = () => chatUI.onReset();
    getElementAndCheck("chatui-send-btn").onclick = () => chatUI.onGenerate();
    chatUI.uiChatInput.onkeypress = (event) => {
      if (event.key === "Enter") {
        chatUI.onGenerate();
      }
    };

    // Try to get previously selected model from local storage
    const savedModelId = ChatUI.getSavedModelId();

    // Get device capabilities
    try {
      const deviceInfo = await chatUI.getDeviceCapabilities(engine);
      if (deviceInfo.restrictModels) {
        chatUI.appendMessage(
          "init",
          "Your device seems to have limited resources, so we restrict the selectable models."
        );
      }

      // Populate and configure model selector
      chatUI.initializeModelSelector(deviceInfo.restrictModels, deviceInfo.maxStorageBufferBindingSize, savedModelId);
      
      return chatUI;
    } catch (err) {
      chatUI.appendMessage("error", `Initialization error: ${err instanceof Error ? err.message : String(err)}`);
      console.error(err);
      return undefined;
    }
  }

  /**
   * Get device capabilities and restrictions
   */
  private async getDeviceCapabilities(engine: webllm.MLCEngineInterface): Promise<{
    restrictModels: boolean;
    maxStorageBufferBindingSize: number;
    gpuVendor: string;
  }> {
    const [maxStorageBufferBindingSize, gpuVendor] = await Promise.all([
      engine.getMaxStorageBufferBindingSize(),
      engine.getGPUVendor(),
    ]);
    
    const restrictModels = 
      (gpuVendor.length !== 0 && MOBILE_GPU_VENDORS.has(gpuVendor.toLowerCase())) || 
      maxStorageBufferBindingSize <= ANDROID_MAX_STORAGE_BUFFER_SIZE;
      
    return { restrictModels, maxStorageBufferBindingSize, gpuVendor };
  }

  /**
   * Initialize and populate the model selector dropdown
   */
  private initializeModelSelector(
    restrictModels: boolean, 
    maxStorageBufferBindingSize: number, 
    savedModelId: string | null
  ): void {
    const modelSelector = getElementAndCheck<HTMLSelectElement>("chatui-select");
    let foundSavedModel = false;
    let lastModelFamily = "";
    
    for (const item of this.config.model_list) {
      // Add separator between different model families
      const currentModelFamily = item.model_id.split("-")[0];
      if (lastModelFamily !== currentModelFamily) {
        if (lastModelFamily !== "") {
          modelSelector.appendChild(document.createElement("hr"));
        }
        lastModelFamily = currentModelFamily;
      }

      // Create option element
      const opt = document.createElement("option");
      opt.value = item.model_id;
      opt.textContent = item.model_id; // Use textContent instead of innerHTML for security

      // Check if this model should be selected
      if (savedModelId && item.model_id === savedModelId) {
        opt.selected = true;
        foundSavedModel = true;
      } else {
        opt.selected = !savedModelId && item === this.config.model_list[0];
      }
      
      // Check if model should be disabled due to device restrictions
      const shouldDisable = this.shouldDisableModel(item, restrictModels, maxStorageBufferBindingSize);
      if (shouldDisable) {
        opt.disabled = true;
        if (opt.selected) {
          opt.selected = false;
          foundSavedModel = false;
        }
      }
      
      modelSelector.appendChild(opt);
    }
    
    // Add final separator
    modelSelector.appendChild(document.createElement("hr"));

    // Set the selected model
    this.selectedModel = modelSelector.value;
    
    // Save initial selection if using default and not found saved model
    if (!foundSavedModel && this.selectedModel) {
      this.saveSelectedModel(this.selectedModel);
    }
    
    // Add change event handler
    modelSelector.onchange = () => this.onSelectChange(modelSelector);
    
    // Sync with sidebar model selector if it exists
    if (this.uiSidebarModelSelect) {
      this.syncSidebarModelSelect(modelSelector);
    }
  }

  /**
   * Sync main model selector with sidebar model selector
   */
  private syncSidebarModelSelect(mainSelector: HTMLSelectElement): void {
    // Clear existing options
    this.uiSidebarModelSelect.innerHTML = "";
    
    // Clone options from main selector
    Array.from(mainSelector.options).forEach(opt => {
      const newOpt = document.createElement("option");
      newOpt.value = opt.value;
      newOpt.textContent = opt.textContent;
      newOpt.disabled = opt.disabled;
      newOpt.selected = opt.selected;
      
      this.uiSidebarModelSelect.appendChild(newOpt);
    });
    
    // Add change event handler to sidebar selector
    this.uiSidebarModelSelect.onchange = () => {
      mainSelector.value = this.uiSidebarModelSelect.value;
      this.onSelectChange(mainSelector);
    };
  }

  /**
   * Determine if a model should be disabled based on device capabilities
   */
  private shouldDisableModel(
    model: any, 
    restrictModels: boolean, 
    maxStorageBufferBindingSize: number
  ): boolean {
    if ((restrictModels && (model.low_resource_required === undefined || !model.low_resource_required)) ||
        (model.buffer_size_required_bytes && 
         maxStorageBufferBindingSize < model.buffer_size_required_bytes)) {
      
      // Allow bypassing restrictions with URL parameter
      const params = new URLSearchParams(location.search);
      return !params.has("bypassRestrictions");
    }
    return false;
  }

  /**
   * Push a task to the execution queue to ensure sequential execution
   */
  private pushTask(task: () => Promise<void>): void {
    this.chatRequestChain = this.chatRequestChain.then(task).catch(err => {
      console.error("Task execution failed:", err);
    });
  }

  /**
   * Handle generate button click
   */
  private async onGenerate(): Promise<void> {
    if (this.requestInProgress) {
      return;
    }
    this.pushTask(async () => {
      await this.asyncGenerate();
    });
  }

  /**
   * Handle model selection change
   */
  private async onSelectChange(modelSelector: HTMLSelectElement): Promise<void> {
    if (this.requestInProgress) {
      this.engine.interruptGenerate();
    }
    
    this.pushTask(async () => {
      await this.engine.resetChat();
      this.resetChatHistory();
      await this.unloadChat();
      this.selectedModel = modelSelector.value;
      this.saveSelectedModel(this.selectedModel);
      await this.asyncInitChat();
    });
  }

  /**
   * Handle reset button click
   */
  private async onReset(): Promise<void> {
    if (this.requestInProgress) {
      this.engine.interruptGenerate();
    }
    this.pushTask(async () => {
      await this.engine.resetChat();
      // Auto-save the current session before resetting
      await this.autoSaveSession();
      this.resetChatHistory();
      await this.updateSessionList();
      this.currentSessionId = null; // Reset session id for new chat
    });
  }

  /**
   * Append a system/assistant message to the chat
   */
  private appendMessage(kind: MessageKind, text: string): void {
    if (kind === "init") {
      text = "[System Initialize] " + text;
    }
    
    if (!this.uiChat) {
      throw Error("Chat UI element not found");
    }
    
    const msg = `
      <div class="msg ${kind}-msg">
        <div class="msg-bubble">
          <div class="msg-text">${text}</div>
        </div>
      </div>
    `;
    this.uiChat.insertAdjacentHTML("beforeend", msg);
    this.scrollChatToBottom();
  }

  /**
   * Append a user message to the chat with special handling for text content
   */
  private appendUserMessage(text: string): void {
    if (!this.uiChat) {
      throw Error("Chat UI element not found");
    }
    
    const msg = `
      <div class="msg right-msg">
        <div class="msg-bubble">
          <div class="msg-text"></div>
        </div>
      </div>
    `;
    this.uiChat.insertAdjacentHTML("beforeend", msg);
    
    // Get message text container and set text content safely
    const msgElement = this.uiChat.lastElementChild?.lastElementChild
      ?.lastElementChild as HTMLElement | undefined;
    
    if (msgElement) {
      msgElement.textContent = text;
    }
    
    this.scrollChatToBottom();
  }

  /**
   * Update the last message of a specific kind in the chat
   */
  private updateLastMessage(kind: MessageKind, text: string): void {
    if (kind === "init") {
      text = "[System Initialize] " + text;
    }
    
    if (!this.uiChat) {
      throw Error("Chat UI element not found");
    }
    
    const matches = this.uiChat.getElementsByClassName(`msg ${kind}-msg`);
    if (matches.length === 0) {
      throw Error(`No ${kind} message found to update`);
    }
    
    const msg = matches[matches.length - 1];
    const msgText = msg.getElementsByClassName("msg-text");
    if (msgText.length !== 1) {
      throw Error("Expected single message text element");
    }
    
    if (msgText[0].innerHTML === text) return;
    
    // Split text by newlines and create elements for each line
    const list = text.split("\n").map(line => {
      const item = document.createElement("div");
      item.textContent = line;
      return item;
    });
    
    msgText[0].innerHTML = "";
    list.forEach(item => msgText[0].append(item));
    
    this.scrollChatToBottom();
  }

  /**
   * Scroll the chat container to the bottom
   */
  private scrollChatToBottom(): void {
    this.uiChat.scrollTo(0, this.uiChat.scrollHeight);
  }

  /**
   * Reset the chat history and clear UI
   */
  private resetChatHistory(): void {
    this.chatHistory = [];
    
    // Clear all message elements
    const clearTags: MessageKind[] = ["left", "right", "init", "error"];
    for (const tag of clearTags) {
      const matches = [...this.uiChat.getElementsByClassName(`msg ${tag}-msg`)];
      for (const item of matches) {
        this.uiChat.removeChild(item);
      }
    }
    
    // Clear info label
    if (this.uiChatInfoLabel) {
      this.uiChatInfoLabel.innerHTML = "";
    }
  }

  /**
   * Initialize the chat model
   */
  private async asyncInitChat(): Promise<void> {
    if (this.chatLoaded) return;
    
    this.requestInProgress = true;
    this.appendMessage("init", "");
    
    // Setup progress callback
    const initProgressCallback = (report: { text: string }) => {
      this.updateLastMessage("init", report.text);
    };
    this.engine.setInitProgressCallback(initProgressCallback);

    try {
      await this.engine.reload(this.selectedModel);
      this.chatLoaded = true;
    } catch (err) {
      this.appendMessage(
        "error", 
        `Initialization error: ${err instanceof Error ? err.message : String(err)}`
      );
      console.error(err);
      await this.unloadChat();
    } finally {
      this.requestInProgress = false;
    }
  }

  /**
   * Unload the chat model
   */
  private async unloadChat(): Promise<void> {
    await this.engine.unload();
    this.chatLoaded = false;
  }

  /**
   * Generate response for user input
   */
  private async asyncGenerate(): Promise<void> {
    // Initialize chat if not already loaded
    await this.asyncInitChat();
    this.requestInProgress = true;
    
    const prompt = this.uiChatInput.value.trim();
    if (!prompt) {
      this.requestInProgress = false;
      return;
    }

    // Update UI for user message
    this.appendUserMessage(prompt);
    this.uiChatInput.value = "";
    this.uiChatInput.setAttribute("placeholder", "Generating...");

    // Add message to chat and prepare for response
    this.appendMessage("left", "");
    this.chatHistory.push({ role: "user", content: prompt });
    await this.autoSaveSession();

    try {
      await this.generateResponse();
    } catch (err) {
      this.appendMessage(
        "error", 
        `Generation error: ${err instanceof Error ? err.message : String(err)}`
      );
      console.error(err);
      await this.unloadChat();
    } finally {
      this.uiChatInput.setAttribute("placeholder", "Enter your message...");
      this.requestInProgress = false;
    }
  }

  /**
   * Generate response from the model
   */
  private async generateResponse(): Promise<void> {
    let curMessage = "";
    let usage: webllm.CompletionUsage | undefined = undefined;
    
    // Create stream completion
    const completion = await this.engine.chat.completions.create({
      stream: true,
      messages: this.chatHistory,
      stream_options: { include_usage: true },
    });
    
    // Process stream chunks
    for await (const chunk of completion) {
      const curDelta = chunk.choices[0]?.delta.content;
      if (curDelta) {
        curMessage += curDelta;
        this.updateLastMessage("left", curMessage);
      }
      
      if (chunk.usage) {
        usage = chunk.usage;
      }
    }
    
    // Update performance metrics if available
    if (usage) {
      this.uiChatInfoLabel.innerHTML = this.formatUsageInfo(usage);
    }
    
    // Get final message and update chat history
    const finalMessage = await this.engine.getMessage();
    this.updateLastMessage("left", finalMessage);
    this.chatHistory.push({ role: "assistant", content: finalMessage });
    await this.autoSaveSession();
  }

  /**
   * Format usage information for display
   */
  private formatUsageInfo(usage: webllm.CompletionUsage): string {
    return `prompt_tokens: ${usage.prompt_tokens}, ` +
      `completion_tokens: ${usage.completion_tokens}, ` +
      `prefill: ${usage.extra.prefill_tokens_per_s.toFixed(4)} tokens/sec, ` +
      `decoding: ${usage.extra.decode_tokens_per_s.toFixed(4)} tokens/sec`;
  }

  // Save the current chat history as a session in IndexedDB.
  private async saveCurrentSession(): Promise<void> {
    if (this.chatHistory.length > 0) {
      try {
        await addSession({ model: this.selectedModel, history: this.chatHistory, timestamp: Date.now() });
      } catch (e) {
        console.error("Failed to save session:", e);
      }
    }
  }

  // New method: auto-save current session in real time.
  private async autoSaveSession(): Promise<void> {
    const sessionData = { model: this.selectedModel, history: this.chatHistory, timestamp: Date.now() };
    if (this.currentSessionId === null) {
      this.currentSessionId = await addSession(sessionData);
    } else {
      await updateSession(this.currentSessionId, sessionData);
    }
  }

  // Update the sidebar with the list of saved chat sessions.
  private async updateSessionList(): Promise<void> {
    try {
      const sessions = await getSessions();
      const listEl = document.getElementById("chat-session-list");
      if (listEl) {
        listEl.innerHTML = "";
        sessions.forEach((session: any) => {
          const li = document.createElement("li");
          li.textContent = new Date(session.timestamp).toLocaleString();
          li.setAttribute("data-session-id", session.id);
          li.onclick = async () => {
            const loaded = await loadSessionFromDB(session.id);
            if (loaded) {
              this.chatHistory = loaded.history;
              this.uiChat.innerHTML = "";
              loaded.history.forEach(msg => {
                const kind = msg.role === "user" ? "right" : "left";
                this.appendMessage(kind, msg.content);
              });
            }
          };
          listEl.appendChild(li);
        });
      }
    } catch (e) {
      console.error("Failed to update session list:", e);
    }
  }
}

/**
 * Initialize the application
 */
async function initializeApp(): Promise<void> {
  const useWebWorker = appConfig.use_web_worker;
  let engine: webllm.MLCEngineInterface;

  // Create appropriate engine based on configuration
  if (useWebWorker) {
    engine = new webllm.WebWorkerMLCEngine(
      new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
      { appConfig, logLevel: "INFO" },
    );
  } else {
    engine = new webllm.MLCEngine({ appConfig });
  }

  // Create and initialize the chat UI
  await ChatUI.CreateAsync(engine);
}

// Start the application
initializeApp().catch(err => {
  console.error("Failed to initialize application:", err);
  document.body.innerHTML = `<div class="error">Failed to initialize application: ${err.message}</div>`;
});
