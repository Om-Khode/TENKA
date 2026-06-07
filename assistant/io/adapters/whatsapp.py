"""
whatsapp.py — WhatsApp adapter for the messaging bridge.

Thin wrapper around neonize. Implements the standard adapter interface.
All neonize-specific code is isolated here — the bridge is generic.

Supported actions:
  read_messages  — Get recent messages (params: chat_name, limit)
  list_chats     — List recent chats (params: limit)
  send_message   — Send a text message (params: phone OR contact_name, text)
  get_contacts   — Get contacts list (params: query — optional name filter)
"""

import logging
import os
import threading
import time

logger = logging.getLogger("whatsapp")

# JID servers that are NOT private chats — used to identify groups, broadcasts, newsletters
_NON_PRIVATE_SERVERS = {"g.us", "broadcast", "newsletter"}

# JID servers for real user contacts — used for contact list filtering only
_USER_CONTACT_SERVERS = {"s.whatsapp.net", "c.us"}


def _jid_to_str(jid) -> str:
    """Extract clean phone number or identifier from a neonize JID object."""
    try:
        if hasattr(jid, 'User') and jid.User:
            return jid.User
        return str(jid)
    except Exception:
        return str(jid)


def _is_private_chat(jid) -> bool:
    """
    Check if a JID represents a private (1-on-1) chat.
    
    Uses a blacklist approach: anything that's NOT a group, broadcast, or
    newsletter is treated as private. This handles LID JIDs which WhatsApp
    now uses inconsistently for private chats.
    """
    try:
        server = getattr(jid, 'Server', '')
        return server not in _NON_PRIVATE_SERVERS
    except Exception:
        return False


def _is_user_contact_jid(jid) -> bool:
    """Check if a JID is a real user contact (for contact list filtering only)."""
    try:
        server = getattr(jid, 'Server', '')
        return server in _USER_CONTACT_SERVERS
    except Exception:
        return False


class Adapter:
    """WhatsApp adapter using neonize."""

    name: str = "whatsapp"

    def __init__(self):
        self._client = None
        self._connected = threading.Event()
        self._messages: list[dict] = []
        self._max_buffer = 1000
        self._lock = threading.Lock()
        self._session_path: str = ""
        self._client_outdated: bool = False

        # Contact name cache for fast notification name resolution
        self._contact_cache: dict[str, str] = {}  # phone -> name
        self._lid_cache: dict[str, str] = {}       # LID User -> name
        self._contact_cache_ts: float = 0.0
        self._CONTACT_CACHE_TTL: float = 60.0  # refresh every 60 seconds

        # Track outgoing messages for notification suppression
        # Maps chat phone/identifier -> timestamp of last outgoing message
        self._last_sent_to: dict[str, float] = {}

    def connect(self, session_path: str) -> None:
        """
        Connect to WhatsApp. BLOCKING — runs in bridge's background thread.
        On first run with no session, neonize prints QR to terminal automatically.
        """
        from neonize.client import NewClient
        from neonize.events import (
            ConnectedEv, MessageEv, HistorySyncEv,
            ClientOutdatedEv, ConnectFailureEv, LoggedOutEv,
        )
        from neonize.proto.waCompanionReg.WAWebProtobufsCompanionReg_pb2 import DeviceProps as _DeviceProps

        self._session_path = session_path
        self._client_outdated = False

        device_props = _DeviceProps()
        device_props.os = "Windows"
        device_props.platformType = _DeviceProps.DESKTOP
        self._client = NewClient(session_path, props=device_props)

        @self._client.event(ConnectedEv)
        def on_connected(client, event):
            logger.info("[WA] Connected to WhatsApp")
            self._client_outdated = False
            self._connected.set()

        @self._client.event(MessageEv)
        def on_message(client, event):
            self._handle_message_event(event)

        @self._client.event(HistorySyncEv)
        def on_history_sync(client, event):
            self._handle_history_sync(event)

        @self._client.event(ClientOutdatedEv)
        def on_client_outdated(client, event):
            logger.error(
                "[WA] WhatsApp rejected this client as outdated (405). "
                "Run: pip install --upgrade neonize"
            )
            self._client_outdated = True
            self._connected.clear()

        @self._client.event(ConnectFailureEv)
        def on_connect_failure(client, event):
            reason = getattr(event, 'Reason', 'unknown')
            logger.error(f"[WA] Connection failed: reason={reason}")
            self._connected.clear()

        @self._client.event(LoggedOutEv)
        def on_logged_out(client, event):
            reason = getattr(event, 'Reason', 'unknown')
            logger.warning(f"[WA] Logged out by WhatsApp: reason={reason}")
            self._connected.clear()
            self._delete_session()

        logger.info(f"[WA] Connecting with session: {session_path}")
        self._client.connect()

    def _delete_session(self) -> None:
        """Delete the session file so the next connect triggers fresh pairing."""
        if self._session_path and os.path.exists(self._session_path):
            try:
                os.remove(self._session_path)
                logger.info(f"[WA] Session deleted: {self._session_path}")
            except Exception as e:
                logger.warning(f"[WA] Failed to delete session: {e}")

    def pair_phone(self, phone: str) -> str:
        """
        Generate a pairing code for phone-number-based linking (alternative to QR).
        Requires connect() to have been called first (WebSocket must be up).
        Returns the 8-character code the user enters in WhatsApp → Linked Devices.
        """
        if not self._client:
            raise ConnectionError("Client not initialized — call connect() first")
        return self._client.PairPhone(phone, show_push_notification=True)

    def _handle_message_event(self, event) -> None:
        """Process a live incoming/outgoing message event."""
        try:
            from ... import config as _cfg
            text = ""
            msg = event.Message
            if msg.conversation:
                text = msg.conversation
            elif hasattr(msg, 'extendedTextMessage') and msg.extendedTextMessage and msg.extendedTextMessage.text:
                text = msg.extendedTextMessage.text

            if not text:
                return

            info = event.Info
            sender = _jid_to_str(info.MessageSource.Sender)
            chat = _jid_to_str(info.MessageSource.Chat)
            is_from_me = info.MessageSource.IsFromMe

            timestamp = None
            try:
                timestamp = info.Timestamp
            except Exception:
                pass

            entry = {
                "sender": sender,
                "chat": chat,
                "text": text,
                "is_from_me": is_from_me,
                "timestamp": timestamp,
            }

            self._add_message(entry)
            logger.debug(f"[WA] Message from {sender}: {text[:50]}")

            # ── Track outgoing messages for suppression ────────────────────
            if is_from_me:
                chat_jid = info.MessageSource.Chat
                chat_phone = _jid_to_str(chat_jid)
                self._last_sent_to[chat_phone] = time.time()
                # Prune old entries (older than 2x suppress window) to prevent unbounded growth
                _max_age = getattr(_cfg, "MESSAGING_SUPPRESS_WINDOW", 300.0) * 2
                _cutoff = time.time() - _max_age
                self._last_sent_to = {k: v for k, v in self._last_sent_to.items() if v > _cutoff}

            # ── Push notification for incoming messages ──────────────────────
            if not is_from_me:
                # Determine chat type from the CHAT JID, not the sender JID.
                # WhatsApp now uses LID (Linked Identity) JIDs for senders even
                # in private chats (Server="lid"), so checking the sender JID
                # would incorrectly classify all messages as non-private.
                # Chat JID: s.whatsapp.net / c.us = private, g.us = group.
                chat_jid = info.MessageSource.Chat
                is_private = _is_private_chat(chat_jid)
                
                # For sender name resolution, use the chat JID's phone number
                # in private chats (since sender may be a LID, not a phone number)
                sender_phone = _jid_to_str(chat_jid) if is_private else sender
                sender_name = self._resolve_sender_name(sender_phone) if sender_phone.isdigit() else sender_phone
                
                # ── Suppress if user recently sent to this contact ─────────
                _suppress_window = getattr(_cfg, "MESSAGING_SUPPRESS_WINDOW", 300.0)
                _last_sent = self._last_sent_to.get(sender_phone, 0.0)
                if time.time() - _last_sent < _suppress_window:
                    logger.debug(f"[WA] Suppressed notification from {sender_name} — active conversation (sent {int(time.time() - _last_sent)}s ago)")
                    return

                import sys as _sys
                bridge_mod = _sys.modules.get("assistant.io.messaging_bridge")
                if not bridge_mod:
                    return
                bridge_mod.push_notification(
                    service="whatsapp",
                    event_type="incoming_message",
                    data={
                        "sender": sender_phone,
                        "sender_name": sender_name,
                        "text": text,
                        "chat": chat,
                        "chat_type": "private" if is_private else "group",
                        "timestamp": timestamp,
                    },
                )
        except Exception as e:
            logger.warning(f"[WA] Unexpected error in message handler: {e}", exc_info=True)

    def _handle_history_sync(self, event) -> None:
        """Process history sync events."""
        try:
            count = 0
            data = event

            conversations = None
            if hasattr(data, 'Data') and hasattr(data.Data, 'Conversations'):
                conversations = data.Data.Conversations
            elif hasattr(data, 'Conversations'):
                conversations = data.Conversations
            elif hasattr(data, 'data') and hasattr(data.data, 'conversations'):
                conversations = data.data.conversations

            if conversations:
                for conv in conversations:
                    try:
                        chat_jid = ""
                        if hasattr(conv, 'ID'):
                            chat_jid = conv.ID
                        elif hasattr(conv, 'Id'):
                            chat_jid = conv.Id
                        elif hasattr(conv, 'id'):
                            chat_jid = conv.id

                        messages = None
                        if hasattr(conv, 'Messages'):
                            messages = conv.Messages
                        elif hasattr(conv, 'messages'):
                            messages = conv.messages

                        if not messages:
                            continue

                        for hist_msg in messages:
                            try:
                                msg_info = None
                                if hasattr(hist_msg, 'Message'):
                                    msg_info = hist_msg.Message
                                elif hasattr(hist_msg, 'message'):
                                    msg_info = hist_msg.message

                                if not msg_info:
                                    continue

                                text = ""
                                actual_msg = None
                                if hasattr(msg_info, 'Message'):
                                    actual_msg = msg_info.Message
                                elif hasattr(msg_info, 'message'):
                                    actual_msg = msg_info.message

                                if actual_msg:
                                    if hasattr(actual_msg, 'conversation') and actual_msg.conversation:
                                        text = actual_msg.conversation
                                    elif (hasattr(actual_msg, 'extendedTextMessage')
                                          and actual_msg.extendedTextMessage
                                          and hasattr(actual_msg.extendedTextMessage, 'text')):
                                        text = actual_msg.extendedTextMessage.text

                                if not text:
                                    continue

                                sender = chat_jid
                                is_from_me = False
                                if hasattr(msg_info, 'Key'):
                                    key = msg_info.Key
                                    if hasattr(key, 'FromMe'):
                                        is_from_me = key.FromMe
                                    if hasattr(key, 'Participant') and key.Participant:
                                        sender = _jid_to_str(key.Participant) if hasattr(key.Participant, 'User') else str(key.Participant)
                                    elif hasattr(key, 'RemoteJID') and key.RemoteJID:
                                        sender = _jid_to_str(key.RemoteJID) if hasattr(key.RemoteJID, 'User') else str(key.RemoteJID)

                                timestamp = None
                                if hasattr(msg_info, 'MessageTimestamp'):
                                    timestamp = msg_info.MessageTimestamp

                                entry = {
                                    "sender": sender,
                                    "chat": str(chat_jid),
                                    "text": text,
                                    "is_from_me": is_from_me,
                                    "timestamp": timestamp,
                                }
                                self._add_message(entry)
                                count += 1
                            except Exception as e:
                                logger.debug(f"[WA] Error parsing history message: {e}")
                                continue
                    except Exception as e:
                        logger.debug(f"[WA] Error parsing conversation: {e}")
                        continue

                if count > 0:
                    logger.info(f"[WA] History sync: captured {count} messages")
                return

            logger.info(f"[WA] HistorySync event received but couldn't parse. "
                       f"Type: {type(event).__name__}, "
                       f"Attrs: {[a for a in dir(event) if not a.startswith('_')][:20]}")

        except Exception as e:
            logger.warning(f"[WA] Unexpected error in history sync: {e}", exc_info=True)

    def _add_message(self, entry: dict) -> None:
        """Thread-safe add message to buffer."""
        with self._lock:
            self._messages.append(entry)
            if len(self._messages) > self._max_buffer:
                self._messages = self._messages[-self._max_buffer:]

    def disconnect(self) -> None:
        """Disconnect from WhatsApp."""
        if self._client:
            try:
                self._client.disconnect()
            except Exception as e:
                logger.debug(f"[WA] Disconnect error: {e}")
        self._connected.clear()
        self._client = None

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def get_client(self):
        return self._client

    # ── Channel protocol ──────────────────────────────────────────────────────

    async def send(self, message: str, recipient: str | None = None) -> bool:
        """Channel-protocol send: thin async wrapper around _send_message."""
        try:
            result = self._send_message(
                {"to": recipient or "", "message": message, "_confirmed": True}
            )
            return result.get("ok", False) if isinstance(result, dict) else False
        except Exception:
            return False

    async def start(self) -> None:
        """Channel-protocol start: connection lifecycle is managed by messaging_bridge."""
        pass

    async def stop(self) -> None:
        """Channel-protocol stop: delegate to disconnect()."""
        self.disconnect()

    def execute(self, action: str, params: dict) -> dict | list | str:
        """
        Execute a WhatsApp action.

        Supported actions:
          read_messages — params: {limit?, chat_name?}
          list_chats    — params: {limit?}
          send_message  — params: {phone? OR contact_name?, text, _confirmed?}
          get_contacts  — params: {query?}
        """
        if not self.is_connected():
            raise ConnectionError("WhatsApp is not connected")

        if action == "read_messages":
            return self._read_messages(params)
        elif action == "list_chats":
            return self._list_chats(params)
        elif action == "send_message":
            return self._send_message(params)
        elif action == "get_contacts":
            return self._get_contacts(params)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _read_messages(self, params: dict) -> list[dict]:
        """Return recent messages from the buffer."""
        limit = int(params.get("limit", 20))
        chat_filter = params.get("chat_name", "").strip().lower()

        with self._lock:
            msgs = list(self._messages)

        if chat_filter:
            msgs = [m for m in msgs if chat_filter in m.get("chat", "").lower()
                     or chat_filter in m.get("sender", "").lower()]

        recent = msgs[-limit:]

        if not recent:
            return [{"sender": "system", "sender_name": "system", "text": "No new messages since I connected. Send or receive a message and try again.", "chat": "", "is_from_me": False, "timestamp": ""}]

        formatted = []
        for m in recent:
            ts_str = self._format_timestamp(m.get("timestamp"))
            phone = m.get("sender", "unknown")
            sender_name = self._resolve_sender_name(phone) if phone != "unknown" else "unknown"
            formatted.append({
                "sender": phone,
                "sender_name": sender_name,
                "chat": m.get("chat", ""),
                "text": m.get("text", ""),
                "is_from_me": m.get("is_from_me", False),
                "timestamp": ts_str,
            })
        return formatted

    def _list_chats(self, params: dict) -> list[dict]:
        """Return unique chats from the message buffer."""
        limit = int(params.get("limit", 20))

        with self._lock:
            msgs = list(self._messages)

        chats: dict[str, dict] = {}
        for m in msgs:
            ts_str = self._format_timestamp(m.get("timestamp"))
            chat_id = m.get("chat", "")
            chats[chat_id] = {
                "chat": chat_id,
                "last_message": m.get("text", "")[:100],
                "last_sender": m.get("sender", ""),
                "timestamp": ts_str,
            }

        sorted_chats = sorted(chats.values(),
                               key=lambda c: c.get("timestamp", ""),
                               reverse=True)
        return sorted_chats[:limit]

    def _send_message(self, params: dict) -> dict | str:
        """
        Send a text message — or return a confirmation dict.

        Two modes:
        1. contact_name or phone provided WITHOUT _confirmed flag:
           → Resolve contact, return needs_confirmation dict. Does NOT send.
        2. phone provided WITH _confirmed=true:
           → Actually send. Called by actions.py after user confirms.

        Group JIDs are always blocked.
        """
        from neonize.utils import build_jid

        phone = params.get("phone", "").strip()
        contact_name = params.get("contact_name", "").strip()
        text = params.get("text", "").strip()
        confirmed = str(params.get("_confirmed", "")).lower() == "true"

        if not text:
            raise ValueError("'text' is required")

        # ── Resolve contact name to phone number ──────────────────────────
        resolved_name = ""
        if not phone and contact_name:
            # Detect phone numbers passed as contact_name (e.g. "9764280339")
            stripped = contact_name.replace("+", "").replace("-", "").replace(" ", "")
            if stripped.isdigit() and 7 <= len(stripped) <= 15:
                phone = contact_name
                logger.info(f"[WA] contact_name '{contact_name}' is a phone number, using directly")
            else:
                matches = self._resolve_contact(contact_name)
                if not matches:
                    raise ValueError(f"No contact found matching '{contact_name}'. Try using a phone number instead.")
                if len(matches) == 1:
                    phone = matches[0]["phone"]
                    resolved_name = matches[0]["name"]
                    logger.info(f"[WA] Resolved '{contact_name}' -> {phone} ({resolved_name})")
                else:
                    if len(matches) > 2:
                        names_list = f"{matches[0]['name']}, {matches[1]['name']} and {len(matches) - 2} others"
                    else:
                        names_list = ", ".join(m["name"] for m in matches)
                    raise ValueError(
                        f"Multiple contacts named '{contact_name}': {names_list}. "
                        f"Which one? Say the full name."
                    )

        if not phone:
            raise ValueError("Either 'phone' or 'contact_name' is required")

        # Clean phone number
        phone = phone.replace("+", "").replace(" ", "").replace("-", "")

        # ── Block group messages ──────────────────────────────────────────
        # Validate by building JID and checking — but for simple check:
        # real phone numbers are 7-15 digits, no dashes after cleaning
        if not phone.isdigit() or len(phone) < 7 or len(phone) > 15:
            raise ValueError("Sending messages to groups is not allowed. I can only send to individual contacts.")

        # ── Confirmation gate ─────────────────────────────────────────────
        if not confirmed:
            display_name = resolved_name or contact_name or phone
            return {
                "needs_confirmation": True,
                "resolved_name": display_name,
                "phone": phone,
                "text": text,
                "service": "whatsapp",
            }

        # ── Actually send (only when _confirmed=true) ─────────────────────
        jid = build_jid(phone)
        self._client.send_message(jid, text)
        logger.info(f"[WA] Message sent to {phone}: {text[:50]}")
        return f"Message sent to {resolved_name or phone}"

    def _get_contacts(self, params: dict) -> list[dict]:
        """Get contacts, optionally filtered by name query."""
        query = params.get("query", "").strip().lower()
        contacts = self._build_contact_list()

        if query:
            contacts = [c for c in contacts if query in c["name"].lower()]

        return contacts[:50]

    # ── Contact Resolution ────────────────────────────────────────────────

    def _resolve_contact(self, name_query: str) -> list[dict]:
        """
        Fuzzy-match a contact name against the contact store + message buffer.

        Priority:
          1. Exact match (case-insensitive) on full name
          2. Substring match — query in name OR name in query
             (both sides must be at least 2 chars to avoid single-letter noise)

        Returns list of {name, phone} dicts.
        """
        contacts = self._build_contact_list()
        query_lower = name_query.lower()

        exact = []
        substring = []

        for c in contacts:
            c_name_lower = c["name"].lower()
            if c_name_lower == query_lower:
                exact.append(c)
            elif len(query_lower) >= 2 and len(c_name_lower) >= 2:
                if query_lower in c_name_lower or c_name_lower in query_lower:
                    substring.append(c)

        return exact if exact else substring

    def _build_contact_list(self) -> list[dict]:
        """
        Build a unified contact list from two sources:
          1. neonize contact store (saved contacts with names)
          2. Message buffer senders (fallback)

        Filters:
          - Only JIDs with Server in {s.whatsapp.net, c.us} (real users)
          - Skips LID, group, newsletter, bot, broadcast JIDs
          - Phone number must be 7-15 digits

        get_all_contacts() returns a
        RepeatedCompositeFieldContainer[Contact].
        Each Contact has:
          .JID       — JID protobuf with .User, .Server
          .Info      — ContactInfo with .FullName, .PushName, .BusinessName, .FirstName
        """
        contacts: dict[str, dict] = {}  # phone -> {name, phone}

        # Source 1: neonize contact store
        try:
            if self._client:
                raw_contacts = self._client.contact.get_all_contacts()
                for contact in raw_contacts:
                    try:
                        jid = contact.JID if hasattr(contact, 'JID') else None
                        if not jid:
                            continue

                        # Filter: only real user JIDs (not groups, newsletters, etc.)
                        if not _is_user_contact_jid(jid):
                            continue

                        phone = _jid_to_str(jid)
                        if not phone or not phone.isdigit() or len(phone) < 7 or len(phone) > 15:
                            continue

                        # Extract name from ContactInfo
                        info = contact.Info if hasattr(contact, 'Info') else None
                        name = ""
                        if info:
                            if info.FullName:
                                name = info.FullName
                            elif info.PushName:
                                name = info.PushName
                            elif info.BusinessName:
                                name = info.BusinessName
                            elif info.FirstName:
                                name = info.FirstName
                        if name:
                            contacts[phone] = {"name": name, "phone": phone}
                    except Exception as e:
                        logger.debug(f"[WA] Error parsing contact entry: {e}")
                        continue
                logger.debug(f"[WA] Contact store: {len(contacts)} named contacts")
        except Exception as e:
            logger.warning(f"[WA] Failed to read contact store: {e}")

        # Source 2: message buffer senders (fallback for contacts not in store)
        with self._lock:
            msgs = list(self._messages)

        for m in msgs:
            sender = m.get("sender", "")
            chat = m.get("chat", "")
            for phone_candidate in (sender, chat):
                if (phone_candidate
                    and phone_candidate.isdigit()
                    and 7 <= len(phone_candidate) <= 15
                    and phone_candidate not in contacts):
                    contacts[phone_candidate] = {"name": phone_candidate, "phone": phone_candidate}

        return list(contacts.values())

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _format_timestamp(ts) -> str:
        """Convert a neonize timestamp to readable format."""
        if not ts:
            return ""
        try:
            ts_val = int(ts) if not isinstance(ts, int) else ts
            if ts_val > 9999999999:
                ts_val = ts_val // 1000
            from datetime import datetime
            return datetime.fromtimestamp(ts_val).strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError, OSError):
            return str(ts)


    def _resolve_sender_name(self, identifier: str) -> str:
        """
        Fast contact name lookup for notifications. Uses cached contact and
        LID maps that refresh every 60 seconds.
        
        Handles both real phone numbers and LID identifiers.
        Returns the contact name, or the phone number, or the raw identifier.
        """
        import time
        now = time.time()
        
        # Refresh caches if stale
        if now - self._contact_cache_ts > self._CONTACT_CACHE_TTL:
            try:
                self._rebuild_caches()
                self._contact_cache_ts = now
            except Exception as e:
                logger.debug(f"[WA] Contact cache refresh failed: {e}")
        
        # Try phone-based lookup first
        if identifier in self._contact_cache:
            return self._contact_cache[identifier]
        
        # Try LID-based lookup
        if identifier in self._lid_cache:
            return self._lid_cache[identifier]
        
        # If it looks like a real phone number, return it as-is
        if identifier.isdigit() and 7 <= len(identifier) <= 15:
            return identifier
        
        # Cross-reference message buffer as last resort
        with self._lock:
            for m in reversed(self._messages):
                if m.get("chat") == identifier or m.get("sender") == identifier:
                    for field in ("sender", "chat"):
                        candidate = m.get(field, "")
                        if (candidate != identifier 
                            and candidate.isdigit() 
                            and 7 <= len(candidate) <= 15):
                            if candidate in self._contact_cache:
                                return self._contact_cache[candidate]
                            return candidate
        
        return identifier
    

    def _rebuild_caches(self) -> None:
        """Rebuild both phone→name and LID→name caches from the contact store."""
        new_contact_cache: dict[str, str] = {}
        new_lid_cache: dict[str, str] = {}

        if not self._client:
            self._contact_cache = new_contact_cache
            self._lid_cache = new_lid_cache
            return

        try:
            raw_contacts = self._client.contact.get_all_contacts()
            for contact in raw_contacts:
                try:
                    jid = contact.JID if hasattr(contact, 'JID') else None
                    if not jid:
                        continue
                    
                    # Extract name
                    info = contact.Info if hasattr(contact, 'Info') else None
                    name = ""
                    if info:
                        if info.FullName:
                            name = info.FullName
                        elif info.PushName:
                            name = info.PushName
                        elif info.BusinessName:
                            name = info.BusinessName
                        elif info.FirstName:
                            name = info.FirstName
                    
                    if not name:
                        continue
                    
                    user = _jid_to_str(jid)
                    server = getattr(jid, 'Server', '')
                    
                    if server in _USER_CONTACT_SERVERS:
                        # Real phone number contact
                        if user.isdigit() and 7 <= len(user) <= 15:
                            new_contact_cache[user] = name
                    elif server == "lid":
                        # LID contact — store the LID User for lookup
                        new_lid_cache[user] = name
                    
                except Exception:
                    continue
            
            logger.debug(f"[WA] Caches rebuilt: {len(new_contact_cache)} phone, {len(new_lid_cache)} LID")
            self._contact_cache = new_contact_cache
            self._lid_cache = new_lid_cache
        except Exception as e:
            logger.debug(f"[WA] Cache rebuild failed: {e}")


# ─── Self-registration ────────────────────────────────────────────────────────
# Import here (after class definition) so the channel_registry import does not
# create a circular dependency at module top.
from ..channels import channel_registry  # noqa: E402
channel_registry.register("whatsapp", Adapter())
