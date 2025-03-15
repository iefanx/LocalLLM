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

// Create database instance
const db = new ChatDatabase();

// Handle messages from main thread
self.onmessage = async (event) => {
  const { action, data, id } = event.data;
  let result: any = null;
  let error = null;
  
  try {
    switch (action) {
      case 'add':
        result = await db.sessions.add(data);
        break;
      case 'getAll':
        result = await db.sessions.toArray();
        break;
      case 'get':
        result = await db.sessions.get(data);
        break;
      case 'update':
        await db.sessions.update(data.id, {
          model: data.session.model,
          history: data.session.history,
          timestamp: data.session.timestamp
        });
        result = data.id;
        break;
      case 'delete':
        await db.sessions.delete(data);
        result = true;
        break;
      case 'deleteAll':
        await db.sessions.clear();
        result = true;
        break;
      case 'getLatest':
        const sessions = await db.sessions.toArray();
        if (sessions && sessions.length > 0) {
          // Sort sessions by timestamp (descending)
          sessions.sort((a, b) => b.timestamp - a.timestamp);
          // Check if id exists before returning
          if (sessions[0].id !== undefined) {
            result = await db.sessions.get(sessions[0].id);
          }
        }
        break;
      default:
        throw new Error(`Unknown action: ${action}`);
    }
  } catch (e) {
    error = e.message || String(e);
    console.error(`DB Worker error (${action}):`, e);
  }
  
  // Send response back to main thread
  self.postMessage({ id, action, result, error });
};

// Log worker initialization
console.log("DB worker initialized");
