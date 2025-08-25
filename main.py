# Add this import at the top of main.py
from .memory_manager import ConversationMemory

# Modify the VoiceAssistant.__init__ method:
class VoiceAssistant:
    def __init__(self):
        self.current_lang = config.lang_default
        self.current_model = config.default_model
        self.sys_prompt = make_system_prompt(self.current_lang)
        self.history: List[Dict[str, str]] = []
        
        # Initialize components
        self.led_manager = LEDManager()
        self.busy = BusyIndicator(self.led_manager, period=0.12)
        self.stt_manager = STTManager()
        
        # Initialize conversation memory
        self.memory = ConversationMemory()
        
        # TTS engine
        self.tts = PiperTTS(
            led_manager=self.led_manager,
            max_chunks=config.max_chunks,
            max_chars=config.max_chars,
            pause_s=config.pause_s,
            prefix_zwsp=True,
        )
        
        # STT backend
        self.stt_mode = config.stt_backend.lower()
        self.stt_obj = None
        self.init_stt()
        
        # Load memory primer on startup
        self._load_memory_primer()

    def _load_memory_primer(self):
        """Load memory primer and enhance system prompt"""
        primer = self.memory.load_or_build_primer()
        if primer:
            self.sys_prompt = f"{self.sys_prompt}\n\n## Previous Context:\n{primer}"
            logger.info("Loaded memory primer into system prompt")

# Add this method to VoiceAssistant class:
    def _enhance_prompt_with_memory(self, user_input: str) -> str:
        """Enhance the prompt with relevant conversation memory"""
        memory_context = self.memory.get_memory_context(user_input)
        
        if not memory_context:
            return self.sys_prompt
        
        enhanced_prompt = f"""{self.sys_prompt}

{memory_context}

Current Instruction: When appropriate, you may choose to remember important information from this conversation. 
Use phrases like "I'll remember that..." or "Noted: ..." when storing important information.
Respond to the user's current query while being aware of our previous conversation context.
"""
        return enhanced_prompt

# Modify the normal chat processing section:
    # Normal chat processing
    user_for_llm = user
    did_inject = ai_status_injection_needed(user)
    
    if did_inject:
        snap = collect_system_snapshot()
        user_for_llm += "\n\nSYSTEM_SNAPSHOT:\n```json\n" + json.dumps(
            snap, ensure_ascii=False, indent=2
        ) + "\n```"
    
    # Enhance prompt with memory context
    enhanced_prompt = self._enhance_prompt_with_memory(user)
    
    self.busy.start("thinking")
    try:
        result = chat_once(
            self.history, 
            enhanced_prompt,  # Use enhanced prompt instead of sys_prompt
            user_for_llm, 
            self.current_model, 
            deterministic=False
        )
        reply = result["reply"]
        
        # Save to memory if the conversation seems important
        self._save_to_memory_if_important(user, reply)
        
        # Keep history bounded
        self.history = result["history"][-config.max_history_length:]
        
        print(f"\nBOT ({self.current_model}): {reply}\n")
        self.tts.speak(reply)

# Add this method to VoiceAssistant class:
    def _save_to_memory_if_important(self, user_input: str, ai_response: str) -> None:
        """Determine if the conversation should be saved to memory"""
        if self.memory.durable_trigger(user_input, ai_response):
            bullets = self.memory.extract_notes(user_input, ai_response, self.history[-6:])
            if bullets:
                self.memory.add_bullets(bullets)
                self.memory.append_notes_md(bullets)
                # Rebuild primer if needed
                self.memory.maybe_rebuild_primer()
                logger.info(f"Saved {len(bullets)} notes to memory")
                
                # Update system prompt with new primer
                primer = self.memory.load_or_build_primer()
                if primer:
                    self.sys_prompt = f"{make_system_prompt(self.current_lang)}\n\n## Previous Context:\n{primer}"

# Add memory commands to handle_command method:
    if low.startswith("/memory"):
        parts = user_input.split()
        if len(parts) > 1 and parts[1] == "clear":
            self.memory.clear_memory()
            # Reset system prompt without memory
            self.sys_prompt = make_system_prompt(self.current_lang)
            print("Conversation memory cleared.")
        elif len(parts) > 1 and parts[1] == "rebuild":
            primer = self.memory._rebuild_primer()
            if primer:
                self.sys_prompt = f"{make_system_prompt(self.current_lang)}\n\n## Previous Context:\n{primer}"
                print("Memory primer rebuilt and loaded.")
            else:
                print("No notes available to rebuild primer.")
        else:
            memory_count = len(self.memory.state.get("notes", []))
            primer = self.memory.state.get("primer", "")
            print(f"Conversation memory: {memory_count} entries")
            if primer:
                print("\nCurrent Primer:")
                print(primer[:200] + "..." if len(primer) > 200 else primer)
            if memory_count > 0:
                print(f"\nLast {min(3, memory_count)} notes:")
                for i, note in enumerate(self.memory.state["notes"][-3:], 1):
                    print(f"  {i}. [{note.get('ts', '')[:10]}] {note.get('text', '')[:60]}...")
        return True