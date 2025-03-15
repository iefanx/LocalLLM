import Dexie from 'dexie';

interface ChatSession {
	id?: number;
	model: string;
	history: any[];
	timestamp: number;
}

class ChatDatabase extends Dexie {
	sessions: Dexie.Table<ChatSession, number>;

	constructor() {
		super("ChatSessionsDB");
		this.version(1).stores({
			sessions: "++id, model, history, timestamp"
		});
		this.sessions = this.table("sessions");
	}
}

const db = new ChatDatabase();

// Create a singleton DB worker
let dbWorker: Worker;
let messageId = 0;
const pendingRequests = new Map<number, { resolve: Function; reject: Function }>();

// Initialize the DB worker
function getDBWorker() {
  if (!dbWorker) {
    dbWorker = new Worker(new URL('./db-worker.ts', import.meta.url), { type: 'module' });
    dbWorker.onmessage = handleWorkerMessage;
    dbWorker.onerror = (error) => {
      console.error('DB Worker error:', error);
    };
  }
  return dbWorker;
}

// Handle messages from the worker
function handleWorkerMessage(event: MessageEvent) {
  const { id, result, error } = event.data;
  const pendingRequest = pendingRequests.get(id);
  
  if (pendingRequest) {
    if (error) {
      pendingRequest.reject(new Error(error));
    } else {
      pendingRequest.resolve(result);
    }
    pendingRequests.delete(id);
  }
}

// Send message to worker and create a promise
function sendToWorker(action: string, data?: any): Promise<any> {
  return new Promise((resolve, reject) => {
    const id = messageId++;
    pendingRequests.set(id, { resolve, reject });
    getDBWorker().postMessage({ action, data, id });
  });
}

// Database API functions
export async function addSession(session: any): Promise<number> {
  return await sendToWorker('add', session);
}

export async function getSessions(): Promise<any[]> {
  return await sendToWorker('getAll');
}

export async function loadSession(id: number): Promise<any | undefined> {
  return await sendToWorker('get', id);
}

export async function updateSession(id: number, session: any): Promise<number> {
  return await sendToWorker('update', { id, session });
}

export async function deleteSession(id: number): Promise<void> {
  await sendToWorker('delete', id);
}

export async function deleteAllSessions(): Promise<void> {
  await sendToWorker('deleteAll');
}

export async function getLatestSession(): Promise<any | null> {
  return await sendToWorker('getLatest');
}
