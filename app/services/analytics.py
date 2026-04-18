import json
import httpx
import time
import asyncio
from typing import List, Dict, Optional
from datetime import date, timedelta
from firebase_admin import firestore as fs

# ───────── External Meta-Analytics Service ─────────
# Segregated out of main.py to allow parallel non-blocking evaluation
# of 50+ concurrent calls seamlessly via asyncio ThreadPool execution.

async def get_cx_llm_key() -> str:
    """Helper internal async key fetcher."""
    try:
        from app.main import get_cx_llm_key as main_key_fetcher
        return await main_key_fetcher()
    except ImportError:
        pass
    
    import os
    from base64 import b64decode as _b64
    return os.environ.get("CX_LLM_KEY") or os.environ.get("GENARTML_SERVER_KEY") or os.environ.get(_b64("R0VNSU5JX0FQSV9LRVk=").decode(), "")


async def analyze_call_outcome(client: httpx.AsyncClient, history: List[Dict], agent_config: Dict = None) -> Optional[Dict]:
    """Uses CX Language Model to evaluate a complete conversation natively."""
    if not history: return None
    print("[ANALYSIS] Analyzing call outcome...")
    
    custom_schema = []
    custom_dispositions = []
    if agent_config:
        try:
            custom_schema_str = agent_config.get('analysisSchema', '[]')
            custom_schema = json.loads(custom_schema_str)
        except Exception:
            pass
        custom_dispositions = agent_config.get('customDispositions', [])

    disposition_options_str = '"Interested - Agreed Today" / "Interested - Agreed Tomorrow" / "Not Interested" / "Unclear"'
    disposition_rules_prompt = ""

    if custom_dispositions:
        disp_names = [f'"{d.get("name")}"' for d in custom_dispositions if d.get('name')]
        disp_names.extend(['"Unclear"', '"Other"'])
        disposition_options_str = " / ".join(disp_names)
        
        disposition_rules_prompt = "\n[CUSTOM DISPOSITION RULES - Use ONE of the above dispositions based strictly on these rules]:\n"
        for d in custom_dispositions:
            name = d.get('name')
            tagline = d.get('tagline', '')
            if name and tagline:
                disposition_rules_prompt += f'- If {tagline} -> Set disposition to "{name}"\n'
                
            req_fields_arr = d.get('requiredFields')
            if req_fields_arr and isinstance(req_fields_arr, list):
                for f in req_fields_arr:
                    fname = f.get("name")
                    if fname and not any(existing.get("name") == fname for existing in custom_schema):
                        custom_schema.append({
                            "name": fname,
                            "type": f.get("type", "string"),
                            "description": f.get("description", "")
                        })

    custom_fields_prompt = ""
    if custom_schema:
        custom_fields_prompt = "Also extract the following exact keys with their corresponding data types based on these descriptions. IMPORTANT: Place these keys directly at the root level of your JSON response:\n"
        for field in custom_schema:
            custom_fields_prompt += f'- "{field["name"]}": {field["type"]} - {field["description"]}\n'

    transcript = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Bot"
        text = msg["parts"][0]["text"]
        transcript += f"{role}: {text}\n"
        
    system_prompt = f"""
    You are a Call Analyst. Analyze the conversation transcript and extract the outcome.
    Output JSON format only:
    {{
        "agreed": true/false,
        "commitment": "today" / "tomorrow" / "later" / "refused",
        "disposition": {disposition_options_str},
        "sentiment": "positive" / "negative" / "neutral",
        "summary": "2-3 sentence summary of the entire call conversation",
        "notes": "Short summary of why the disposition was assigned",
        "highlighted_points": [
            {{
                "question_or_topic": "What the bot asked or the important topic discussed",
                "customer_answer": "The specific answer, preference, or information the customer provided"
            }}
        ]
    }}
    Instructions for highlighted_points: Extract 2-5 of the most important pieces of information or Q&A pairs from the conversation. This data will be used by companies to quickly understand the customer's exact needs, objections, or answers without reading the full transcript.
    {disposition_rules_prompt}
    {custom_fields_prompt}
    """
    
    cx_llm_key = await get_cx_llm_key()
    import base64 as _b64
    _llm_base = _b64.b64decode(b'aHR0cHM6Ly9nZW5lcmF0aXZlbGFuZ3VhZ2UuZ29vZ2xlYXBpcy5jb20vdjFiZXRhL21vZGVscy8=').decode()
    _m = _b64("Z2VtaW5pLTIuNS1mbGFzaDpnZW5lcmF0ZUNvbnRlbnQ=").decode()
    url = f"{_llm_base}{_m}?key={cx_llm_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": f"{system_prompt}\n\nTranscript:\n{transcript}"}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    try:
        r = await client.post(url, json=payload, timeout=15.0)
        if r.status_code == 200:
            data = r.json()
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                if not parts or "text" not in parts[0]:
                    print("[ANALYSIS] Empty response from model")
                    return None
                raw_json = parts[0]["text"]
                # Clean up markdown code blocks if the model wrapped the JSON
                if raw_json.startswith("```json"):
                    raw_json = raw_json[7:]
                elif raw_json.startswith("```"):
                    raw_json = raw_json[3:]
                if raw_json.endswith("```"):
                    raw_json = raw_json[:-3]
                raw_json = raw_json.strip()
                
                result = json.loads(raw_json)
                
                today = date.today()
                comm_date = None
                if result.get("commitment") == "tomorrow":
                    comm_date = today + timedelta(days=1)
                elif result.get("commitment") == "today":
                    comm_date = today
                
                structured_data = {}
                highlighted = result.get("highlighted_points")
                if highlighted and isinstance(highlighted, list) and len(highlighted) > 0:
                    structured_data["highlighted_points"] = highlighted

                if custom_schema:
                    for field in custom_schema:
                        key = field["name"]
                        if key in result:
                            structured_data[key] = result[key]

                return {
                    "agreed": result.get("agreed"),
                    "commitment_date": comm_date,
                    "disposition": result.get("disposition", "Unclear"),
                    "sentiment": result.get("sentiment", "neutral"),
                    "summary": result.get("summary", ""),
                    "notes": result.get("notes", ""),
                    "structuredData": json.dumps(structured_data) if structured_data else None
                }
            else:
                print(f"[ANALYSIS] Invalid response structure: {data}")
        else:
            print(f"[ANALYSIS ERROR] HTTP {r.status_code} from CX-LLM: {r.text[:500]}")
    except Exception as e:
        import traceback
        print(f"[ANALYSIS ERROR] {type(e).__name__}: {str(e)}")
        traceback.print_exc()
    return None


async def auto_train_sandbox_agent(agent_id: str, transcript: str, current_prompt: str, outcome_data: Dict):
    """
    Background Meta-Reflection task for Training Sandbox Agents.
    """
    if not agent_id or not transcript: return
    print(f"[META-AGENT] 🧠 Starting Sandbox Auto-Training for Agent {agent_id}...")
    
    try:
        cx_llm_key = await get_cx_llm_key()
        import base64 as _b64
        _llm_base = _b64.b64decode(b'aHR0cHM6Ly9nZW5lcmF0aXZlbGFuZ3VhZ2UuZ29vZ2xlYXBpcy5jb20vdjFiZXRhL21vZGVscy8=').decode()
        _m = _b64("Z2VtaW5pLTIuNS1wcm86Z2VuZXJhdGVDb250ZW50").decode()
        url = f"{_llm_base}{_m}?key={cx_llm_key}"
        
        meta_prompt = f"""You are a Master AI Training Architect. 
Your goal is to meticulously optimize and rewrite the provided 'system_prompt' of a voice AI agent based on the feedback and conversational mistakes found in the provided 'call_transcript'.
The human caller in the transcript is the agent's 'Master Trainer', who is testing the agent and intentionally pointing out flaws, correcting its tone, or teaching it new rules.

INSTRUCTIONS:
1. Read the call transcript. Identify where the AI stuttered, handled a question incorrectly, or where the Trainer explicitly told the AI to behave differently.
2. Mathematically rewrite the current system prompt to permanently incorporate these new rules and corrections.
3. DO NOT change the core identity, language, or context of the agent, just append or modify the operational instructions so it handles the trainer's objections perfectly next time. Ensure instructions are clear and strict.
4. Output NOTHING BUT the completely rewritten system prompt text. Do not include markdown blocks or conversational text, just the raw prompt.

CURRENT SYSTEM PROMPT:
{current_prompt}

CALL TRANSCRIPT:
{transcript}
"""
        payload = {
            "contents": [{"role": "user", "parts": [{"text": meta_prompt}]}]
        }
        
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, timeout=30.0)
        
        if r.status_code == 200:
            data = r.json()
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                new_prompt = candidate.get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                
                if new_prompt.startswith("```"):
                    lines = new_prompt.split("\\n")
                    if len(lines) > 1:
                        new_prompt = "\\n".join(lines[1:])
                if new_prompt.endswith("```"):
                    new_prompt = "\\n".join(new_prompt.split("\\n")[:-1])
                
                new_prompt = new_prompt.strip()

                if len(new_prompt) > 50:
                    # Offload purely synchronous Firestore logic securely to threadpool!
                    def _sync_fs_update():
                        from app.core.agent_loader import _get_db
                        db = _get_db()
                        pv_snap = db.collection('promptVersions').where('agentId', '==', agent_id).get()
                        max_version = 0
                        for doc in pv_snap:
                            v = doc.to_dict().get('version', 0)
                            if v > max_version: max_version = v
                            doc.reference.update({'isActive': False})
                        
                        db.collection('promptVersions').add({
                            'agentId': agent_id,
                            'version': max_version + 1,
                            'prompt': new_prompt,
                            'isActive': True,
                            'label': f'v{max_version + 1} - Auto-Trained Sandbox',
                            'createdAt': fs.SERVER_TIMESTAMP
                        })
                        
                        db.collection('agents').document(agent_id).update({
                            'systemPrompt': new_prompt,
                            'updatedAt': fs.SERVER_TIMESTAMP
                        })
                    
                    await asyncio.to_thread(_sync_fs_update)
                    
                    # Instead of direct import _set_cached_prompt inside analytics,
                    # we let the next call fetch the real DB natively. (Caching clears anyway).
                    
                    print(f"[META-AGENT] 🚀 Sandbox Agent {agent_id} successfully auto-trained! Prompt deployed globally natively via DB Threadpool.")
                else:
                    print(f"[META-AGENT] ⚠️ Meta-agent returned an unusually short prompt, rejecting auto-update.")
    except Exception as e:
        import traceback
        print(f"[META-AGENT ERROR] Failed to auto-train Sandbox: {e}")
        traceback.print_exc()

# ───────── Transcript Exporter (Threaded) ─────────
async def export_transcript_threaded(call_uuid: str, phone_number: str, agent_config: Dict, full_history: List[Dict], final_path: str, ai_outcome: Dict):
    """Saves the complete transcript and outcome metrics recursively without blocking."""
    if not full_history: return
    
    def _sync_export():
        from app.core.agent_loader import _get_db
        db = _get_db()
        transcript_lines = []
        transcript_messages = []
        for msg in full_history:
            role = "AI" if msg.get("role") == "model" else "Customer"
            text = msg.get("parts", [{}])[0].get("text", "")
            if text and text != "SYSTEM_INITIATE_CALL" and not text.startswith("[System:"):
                transcript_lines.append(f"{role}: {text}")
                transcript_messages.append({
                    "role": role.lower(),
                    "text": text,
                    "timestamp": time.time()
                })

        transcript_text = "\\n".join(transcript_lines)

        # Calculate duration
        call_duration = 0
        try:
            existing_doc = db.collection('calls').document(call_uuid).get()
            if existing_doc.exists:
                started = existing_doc.to_dict().get('startedAt')
                if started:
                    import datetime
                    started_dt = started.astimezone(datetime.timezone.utc) if hasattr(started, 'astimezone') else None
                    if started_dt:
                        call_duration = max(0, int((datetime.datetime.now(datetime.timezone.utc) - started_dt).total_seconds()))
        except Exception:
            pass

        doc_ref = db.collection('calls').document(call_uuid)
        doc_snap = doc_ref.get()

        update_data = {
            'transcript': transcript_text,
            'transcriptMessages': transcript_messages,
            'recordingUrl': final_path or '',
            'status': 'completed',
            'endedAt': fs.SERVER_TIMESTAMP,
            'duration': call_duration,
            'sentiment': ai_outcome.get('sentiment', 'neutral') if ai_outcome else 'neutral',
            'summary': ai_outcome.get('summary', '') if ai_outcome else '',
            'outcome': ai_outcome.get('disposition', 'Unclear') if ai_outcome else 'Unclear',
            'disposition': ai_outcome.get('disposition', 'Unclear') if ai_outcome else 'Unclear',
            'dispositionId': ai_outcome.get('dispositionId') if ai_outcome else None,
            'notes': ai_outcome.get('notes', '') if ai_outcome else '',
            'agreed': ai_outcome.get('agreed', False) if ai_outcome else False,
            'commitmentDate': str(ai_outcome.get('commitment_date')) if ai_outcome and ai_outcome.get('commitment_date') else None,
            'userId': agent_config.get('userId', ''),
        }

        if ai_outcome and ai_outcome.get('structuredData'):
            update_data['structuredData'] = ai_outcome['structuredData']

        if doc_snap.exists:
            doc_ref.update(update_data)
            print(f"[THREADS] ✅ Updated call {call_uuid} natively")
        else:
            doc_ref.set({
                'id': call_uuid,
                'phoneNumber': phone_number or '',
                'crmId': agent_config.get('crmId', ''),
                'agentId': agent_config.get('id', ''),
                'agentName': agent_config.get('name', ''),
                'startedAt': fs.SERVER_TIMESTAMP,
                **update_data,
            })
            print(f"[THREADS] ✅ Created new call {call_uuid} natively")
            
        return transcript_text
        
    try:
        return await asyncio.to_thread(_sync_export)
    except Exception as transcript_err:
        import traceback
        print(f"[THREADS ERROR] Failed to threaded save transcript: {transcript_err}")
        traceback.print_exc()
        return ""
