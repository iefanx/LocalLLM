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

// Replace Dixie functions with Dexie API
export async function addSession(session: ChatSession): Promise<number> {
	return await db.sessions.add(session);
}

export async function getSessions(): Promise<ChatSession[]> {
	return await db.sessions.toArray();
}

export async function loadSession(id: number): Promise<ChatSession | undefined> {
	return await db.sessions.get(id);
}

// Added updateSession to update an existing session record
export async function updateSession(id: number, session: ChatSession): Promise<number> {
	await db.sessions.update(id, { model: session.model, history: session.history, timestamp: session.timestamp });
	return id;
}
