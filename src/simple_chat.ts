import appConfig from "./app-config";
import * as webllm from "@mlc-ai/web-llm";
import { refreshIcons } from "./icons"; // Import the icon refresh function
import { addSession, getSessions, loadSession as loadSessionFromDB, updateSession, getLatestSession, deleteSession, deleteAllSessions } from "./chat-session-db";
import { marked } from "marked";
// Remove direct import and use the globally available KaTeX from CDN
// import renderMathInElement from "/katex/contrib/auto-render.min.js";

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
  private readonly uiSendButton: HTMLButtonElement;  // New property for send/stop button
  
  private selectedModel: string = "gemma-3-1b-it-q4f16_1-MLC"; // Default model
  private chatLoaded = false;
  private requestInProgress = false;
  private chatHistory: webllm.ChatCompletionMessageParam[] = [];
  
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
    this.uiSendButton = getElementAndCheck("chatui-send-btn") as HTMLButtonElement; // Save reference
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
    
    // Add new chat button event handler
    try {
      const newChatBtn = getElementAndCheck("chatui-new-chat-btn");
      newChatBtn.onclick = () => chatUI.createNewSessionWithoutClosingCurrent();
    } catch (e) {
      console.warn("New chat button not found in the interface");
    }
    
    chatUI.uiChatInput.onkeypress = (event) => {
      if (event.key === "Enter") {
        chatUI.onGenerate();
      }
    };

    // Register beforeunload event to save session when closing/refreshing
    window.addEventListener('beforeunload', () => {
      if (chatUI.chatHistory.length > 0) {
        chatUI.autoSaveSession();
      }
    });

    // Always use the default model.
    chatUI.selectedModel = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC";
    
    try {
      const deviceInfo = await chatUI.getDeviceCapabilities(engine);
      if (deviceInfo.restrictModels) {
        chatUI.appendMessage(
          "init",
          "Your device seems to have limited resources, so we restrict the selectable models."
        );
      }
      
      // Automatically initialize the model when the app opens
      await chatUI.asyncInitChat();
      
      // Load the most recent chat session if it exists
      await chatUI.loadLatestSession();
      
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
      // Stop generation if already in progress
      this.handleInterrupt();
      return;
    }
    this.pushTask(async () => {
      await this.asyncGenerate();
    });
  }

  /**
   * Handle interruption of text generation
   */
  private handleInterrupt(): void {
    this.engine.interruptGenerate();
    this.uiSendButton.innerHTML = '<i data-lucide="send"></i>';
    this.uiChatInput.setAttribute("placeholder", "Enter your message...");
    this.requestInProgress = false;
    refreshIcons(); // Update icon rendering
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
      this.resetChatHistory();
      await this.updateSessionList();
      this.currentSessionId = null; // Reset session id for new chat
    });
  }

  /**
   * Process message content with Markdown and LaTeX
   */
  private processMessageContent(content: string): string {
    // First parse the markdown - ensure it returns string not Promise<string>
    const htmlContent = marked.parse(content, { async: false }) as string;
    
    // Create a temporary div to hold the content
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = htmlContent;
    
    // Use the globally available renderMathInElement from KaTeX CDN
    if (typeof window.renderMathInElement === 'function') {
      window.renderMathInElement(tempDiv, {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false},
          {left: '\\(', right: '\\)', display: false},
          {left: '\\[', right: '\\]', display: true}
        ],
        throwOnError: false
      });
    } else {
      console.warn("KaTeX renderMathInElement is not available");
    }
    
    return tempDiv.innerHTML;
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
    
    // Process with Markdown and LaTeX if it's an assistant (left) message
    const processedText = (kind === "left") ? this.processMessageContent(text) : text;
    
    const msg = `
      <div class="msg ${kind}-msg">
        <div class="msg-bubble">
          <div class="msg-text">${processedText}</div>
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
    
    // Process with Markdown and LaTeX if it's an assistant (left) message
    if (kind === "left") {
      msgText[0].innerHTML = this.processMessageContent(text);
    } else {
      // Split text by newlines and create elements for each line
      const list = text.split("\n").map(line => {
        const item = document.createElement("div");
        item.textContent = line;
        return item;
      });
      
      msgText[0].innerHTML = "";
      list.forEach(item => msgText[0].append(item));
    }
    
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
    this.appendMessage("init", ""); // initial progress message
    
    // Setup progress callback
    const initProgressCallback = (report: { text: string }) => {
      this.updateLastMessage("init", report.text);
    };
    this.engine.setInitProgressCallback(initProgressCallback);

    try {
      await this.engine.reload(this.selectedModel);
      this.chatLoaded = true;
      // Update initial message with final notice for the user
      this.updateLastMessage("init",
        "Model loaded locally. You can now converse with it offline. Your conversation remains completely private, running entirely on your device.");
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
    this.removeInitialMessage(); // Remove initial message when chatting starts
    this.requestInProgress = true;
    // Change send button to stop button and refresh icons
    this.uiSendButton.innerHTML = '<i data-lucide="circle-pause"></i>';
    refreshIcons(); // Refresh dynamic icon
    const prompt = this.uiChatInput.value.trim();
    if (!prompt) {
      this.requestInProgress = false;
      this.uiSendButton.innerHTML = '<i data-lucide="send"></i>';
      refreshIcons();
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
      // Restore send button icon and refresh icons
      this.uiSendButton.innerHTML = '<i data-lucide="send"></i>';
      refreshIcons();
    }
  }

  // New method to remove initial system message
  private removeInitialMessage(): void {
    const initMsgs = this.uiChat.getElementsByClassName("msg init-msg");
    while (initMsgs.length > 0) {
      this.uiChat.removeChild(initMsgs[0]);
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
    // Don't save empty sessions
    if (this.chatHistory.length === 0) {
      return;
    }
    
    try {
      const sessionData = {
        history: this.chatHistory,
        model: this.selectedModel,
        timestamp: new Date().getTime()
      };
      
      if (this.currentSessionId === null) {
        // Create new session
        const id = await addSession(sessionData);
        this.currentSessionId = id;
        
        // Save as latest session for next app load
        localStorage.setItem('latest-session-id', String(id));
      } else {
        // Update existing session
        await updateSession(this.currentSessionId, sessionData);
        
        // Save as latest session for next app load only if it's not empty
        if (this.chatHistory.length > 0) {
          localStorage.setItem('latest-session-id', String(this.currentSessionId));
        }
      }
      
      await this.updateSessionList();
    } catch (err) {
      console.error("Error saving chat session:", err);
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
      
      // Sort sessions by timestamp (most recent first)
      sessions.sort((a, b) => b.timestamp - a.timestamp);
      
      const listEl = document.getElementById("chat-session-list");
      if (listEl) {
        listEl.innerHTML = "";
        
        // Add session management options menu
        const menuContainer = document.createElement("div");
        menuContainer.className = "session-menu-container";
        
        // Create session options menu
        const optionsMenu = document.createElement("div");
        optionsMenu.className = "session-options-menu";
        
        // Option 1: Create new chat
        const newChatOption = this.createMenuOption(
          "New Chat", 
          "plus-circle", 
          () => this.createNewChat()
        );
        
        // Option 2: Reset current session
        const resetOption = this.createMenuOption(
          "Reset Current Chat", 
          "refresh-cw", 
          () => this.confirmResetCurrentChat()
        );
        
        // Option 3: Delete current session
        const deleteSessionOption = this.createMenuOption(
          "Delete Current Session", 
          "trash-2", 
          () => this.confirmDeleteCurrentSession()
        );
        
        // Option 4: Delete all sessions
        const deleteAllOption = this.createMenuOption(
          "Delete All Sessions", 
          "trash", 
          () => this.confirmDeleteAllSessions()
        );
        
        // Add all options to menu
        optionsMenu.appendChild(newChatOption);
        optionsMenu.appendChild(resetOption);
        optionsMenu.appendChild(deleteSessionOption);
        optionsMenu.appendChild(deleteAllOption);
        
        menuContainer.appendChild(optionsMenu);
        listEl.appendChild(menuContainer);
        
        // Add sessions list heading if sessions exist
        if (sessions.length > 0) {
          const sessionListHeader = document.createElement("div");
          sessionListHeader.className = "session-list-header";
          sessionListHeader.textContent = "Previous Conversations";
          listEl.appendChild(sessionListHeader);
        }
        
        // Add each session to the list (already sorted)
        sessions.forEach((session: any) => {
          // Create list item container
          const li = document.createElement("li");
          li.className = "chat-session-item";
          
          // Highlight current session
          if (this.currentSessionId === session.id) {
            li.classList.add("active");
          }
          
          li.setAttribute("data-session-id", session.id);
          
          // Create session content wrapper
          const contentWrapper = document.createElement("div");
          contentWrapper.className = "session-content";
          
          // Find first user message
          const firstUserMessage = session.history.find((msg: any) => msg.role === "user");
          const messagePreview = firstUserMessage 
            ? this.truncateText(typeof firstUserMessage.content === 'string' ? firstUserMessage.content : "No content", 40)
            : "Empty conversation";
            
          // Create preview element
          const preview = document.createElement("div");
          preview.className = "session-preview";
          preview.textContent = messagePreview;
          
          // Create timestamp element
          const timestamp = document.createElement("div");
          timestamp.className = "session-timestamp";
          timestamp.textContent = new Date(session.timestamp).toLocaleString();
          
          // Create delete button
          const deleteBtn = document.createElement("button");
          deleteBtn.className = "session-delete-btn";
          deleteBtn.innerHTML = "×"; // × character for delete button
          deleteBtn.onclick = (e) => {
            e.stopPropagation(); // Prevent session loading when clicking delete
            this.confirmDeleteSession(session.id);
          };
          
          // Add elements to wrapper
          contentWrapper.appendChild(preview);
          contentWrapper.appendChild(timestamp);
          li.appendChild(contentWrapper);
          li.appendChild(deleteBtn);
          
          // Set click handler for loading the session
          contentWrapper.onclick = async () => {
            await this.loadSession(session.id);
          };
          
          listEl.appendChild(li);
        });
        
        // Refresh icons for the menu items
        refreshIcons();
      }
    } catch (e) {
      console.error("Failed to update session list:", e);
    }
  }
  
  /**
   * Helper method to create a menu option
   */
  private createMenuOption(label: string, iconName: string, onClick: () => void): HTMLDivElement {
    const option = document.createElement("div");
    option.className = "session-menu-option";
    
    const icon = document.createElement("span");
    icon.className = "session-menu-icon";
    icon.innerHTML = `<i data-lucide="${iconName}"></i>`;
    
    const text = document.createElement("span");
    text.className = "session-menu-text";
    text.textContent = label;
    
    option.appendChild(icon);
    option.appendChild(text);
    option.onclick = onClick;
    
    return option;
  }
  
  /**
   * Creates a new chat session without closing the current one
   */
  private async createNewSessionWithoutClosingCurrent(): Promise<void> {
    this.pushTask(async () => {
      // Create a new session while keeping the current one
      this.resetChatHistory(); // Clear UI
      this.currentSessionId = null; // Reset session ID for new chat
      await this.engine.resetChat(); // Reset the engine
      await this.updateSessionList();
      
      // Clear the "latest session" from localStorage to ensure a fresh start next time
      localStorage.removeItem('latest-session-id');
      
      // Show initialization message for new chat
      this.appendMessage("init", "Started a new conversation. Your previous chat is saved in the sidebar.");
    });
  }

  /**
   * Create a new chat session (replacing current)
   */
  private createNewChat(): void {
    this.pushTask(async () => {
      await this.engine.resetChat();
      this.resetChatHistory();
      this.currentSessionId = null;
      await this.updateSessionList();
      
      // Clear the "latest session" from localStorage to ensure a fresh start next time
      localStorage.removeItem('latest-session-id');
    });
  }
  
  /**
   * Show confirmation dialog before resetting current chat
   */
  private confirmResetCurrentChat(): void {
    if (confirm("Are you sure you want to clear all messages in the current conversation?")) {
      this.pushTask(async () => {
        await this.engine.resetChat();
        // Keep the session ID but clear the history
        if (this.currentSessionId !== null) {
          this.chatHistory = [];
          await updateSession(this.currentSessionId, { 
            model: this.selectedModel, 
            history: this.chatHistory, 
            timestamp: Date.now() 
          });
        }
        this.resetChatHistory();
        await this.updateSessionList();
      });
    }
  }
  
  /**
   * Show confirmation dialog before deleting current session
   */
  private confirmDeleteCurrentSession(): void {
    if (this.currentSessionId === null) {
      alert("No current session to delete.");
      return;
    }
    
    if (confirm("Are you sure you want to delete the current conversation?")) {
      this.deleteSession(this.currentSessionId);
    }
  }

  /**
   * Truncate text and add ellipsis if too long
   */
  private truncateText(text: string, maxLength: number): string {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + "...";
  }
  
  /**
   * Load a specific chat session
   */
  private async loadSession(sessionId: number): Promise<void> {
    try {
      const loaded = await loadSessionFromDB(sessionId);
      if (loaded) {
        // Update current session and history
        this.currentSessionId = loaded.id ?? null;
        this.chatHistory = loaded.history;
        this.selectedModel = loaded.model || this.selectedModel;
        
        // Clear chat UI and display loaded messages
        this.uiChat.innerHTML = "";
        loaded.history.forEach(msg => {
          const kind = msg.role === "user" ? "right" : "left";
          this.appendMessage(kind, typeof msg.content === 'string' ? msg.content : String(msg.content || ''));
        });
      }
    } catch (err) {
      console.error("Error loading chat session:", err);
    }
  }
  
  /**
   * Show confirmation dialog before deleting a session
   */
  private confirmDeleteSession(sessionId: number): void {
    if (confirm("Are you sure you want to delete this conversation?")) {
      this.deleteSession(sessionId);
    }
  }
  
  /**
   * Delete a specific chat session
   */
  private async deleteSession(sessionId: number): Promise<void> {
    try {
      await deleteSession(sessionId);
      
      // If we deleted the current session, reset the UI
      if (this.currentSessionId === sessionId) {
        this.resetChatHistory();
        this.currentSessionId = null;
        await this.engine.resetChat();
      }
      
      // If we're deleting what was saved as the latest session, remove that reference
      const latestSessionId = localStorage.getItem('latest-session-id');
      if (latestSessionId === String(sessionId)) {
        localStorage.removeItem('latest-session-id');
      }
      
      await this.updateSessionList();
    } catch (err) {
      console.error("Error deleting session:", err);
      this.appendMessage("error", "Failed to delete session: " + err.message);
    }
  }
  
  /**
   * Show confirmation dialog before deleting all sessions
   */
  private confirmDeleteAllSessions(): void {
    if (confirm("Are you sure you want to delete ALL conversations? This cannot be undone.")) {
      this.deleteAllSessions();
    }
  }
  
  /**
   * Delete all chat sessions
   */
  private async deleteAllSessions(): Promise<void> {
    try {
      await deleteAllSessions();
      
      // Reset current UI state
      this.resetChatHistory();
      this.currentSessionId = null;
      
      // Update the session list
      await this.updateSessionList();
    } catch (err) {
      console.error("Error deleting all sessions:", err);
    }
  }

  /**
   * Load the most recent chat session if available
   */
  private async loadLatestSession(): Promise<void> {
    try {
      // First check if we have a specific session ID stored
      const latestSessionId = localStorage.getItem('latest-session-id');
      
      // If we have a stored session ID, try to load it specifically
      if (latestSessionId) {
        const session = await loadSessionFromDB(Number(latestSessionId));
        if (session && session.history.length > 0) {
          this.currentSessionId = session.id;
          this.chatHistory = session.history;
          this.selectedModel = session.model || this.selectedModel;
          
          // Display the chat history in UI
          this.uiChat.innerHTML = ""; // Clear any initial messages
          this.chatHistory.forEach(msg => {
            const kind = msg.role === "user" ? "right" : "left";
            this.appendMessage(kind, typeof msg.content === 'string' ? msg.content : String(msg.content || ''));
          });
          
          // Update session list in sidebar if present
          await this.updateSessionList();
          return; // Successfully loaded session, exit function
        }
      }
      
      // If no valid session was loaded, get the latest session from database
      const latestSession = await getLatestSession();
      if (latestSession && latestSession.history.length > 0) {
        this.currentSessionId = latestSession.id;
        this.chatHistory = latestSession.history;
        this.selectedModel = latestSession.model || this.selectedModel;
        
        // Display the chat history in UI
        this.uiChat.innerHTML = ""; // Clear any initial messages
        this.chatHistory.forEach(msg => {
          const kind = msg.role === "user" ? "right" : "left";
          this.appendMessage(kind, typeof msg.content === 'string' ? msg.content : String(msg.content || ''));
        });
        
        // Update session list in sidebar if present
        await this.updateSessionList();
        
        // Save this as the latest session ID
        localStorage.setItem('latest-session-id', String(latestSession.id));
      }
    } catch (err) {
      console.error("Error loading chat session:", err);
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
