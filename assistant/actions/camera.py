"""Camera and face recognition handlers: camera_look, meet_face,
recognize_face, forget_face, and their pending state handlers."""

import logging

from .. import config
from .registry import tool_registry
from .responses import personality_say

logger = logging.getLogger("actions")


async def handle_pending_camera_settings(text: str) -> str | None:
    import assistant.actions as _act

    if _act.pending_camera_settings.payload is None:
        return None

    lowered = text.strip().lower()

    is_yes = any(w in lowered for w in (
        "yes", "yeah", "yep", "sure", "ok", "okay", "please",
        "go ahead", "open it", "open", "do it", "show me",
    ))
    is_no = any(w in lowered for w in (
        "no", "nope", "nah", "skip", "cancel", "don't", "dont",
        "never mind", "forget it",
    ))

    if is_yes:
        import os
        _act.pending_camera_settings.clear()
        try:
            os.startfile("ms-settings:privacy-webcam")
            return (
                "I've opened camera settings. Under 'Let desktop apps access your "
                "camera', make sure it's turned on. Then say 'look at me' again and I'll try."
            )
        except Exception as e:
            logger.error(f"[CAMERA] Failed to open settings: {e}")
            return (
                "I couldn't open settings automatically. Go to Windows Settings, "
                "then Privacy, then Camera, and make sure desktop app access is enabled."
            )

    if is_no:
        _act.pending_camera_settings.clear()
        return "Alright, just say 'look at me' whenever you're ready to try again."

    _act.pending_camera_settings.touch()
    return (
        "Just checking — did you want me to open camera settings? Say yes or no."
    )


@tool_registry.decorator("camera_look")
async def handle_camera_look(params: dict, llm_response: str, bridge=None) -> str:
    import assistant.actions as _act
    from ..io import camera
    from .. import faces
    from .. import llm as llm_module
    from .. import config as _config
    import face_recognition as fr

    if not _config.CAMERA_ENABLED:
        return (
            "My camera is currently disabled. "
            "Type /set camera_enabled true to turn it on."
        )

    if _act.pending_camera_settings.active:
        _act.pending_camera_settings.clear()

    if bridge:
        await bridge.send_command("play_animation", name="thinking")

    visual_question = (
        llm_response.strip()
        if llm_response.strip()
        else "Describe what you see in this image."
    )

    frame_rgb = camera.capture_camera_frame_numpy(
        camera_index=_config.CAMERA_INDEX
    )

    if frame_rgb is None:
        _act.pending_camera_settings.set({"waiting": True})
        return (
            "I couldn't access the camera. Want me to open Windows camera "
            "settings so you can check the permissions?"
        )

    recognized_name: str | None = None
    try:
        locations = fr.face_locations(
            frame_rgb, number_of_times_to_upsample=2
        )
        if locations:
            encodings = fr.face_encodings(frame_rgb, locations)
            if encodings:
                matches = faces.find_faces(
                    encodings,
                    tolerance=_config.FACE_RECOGNITION_TOLERANCE
                )
                recognized_names = [m for m in matches if m is not None]
                if recognized_names:
                    recognized_name = recognized_names[0]
    except Exception as e:
        logger.warning(f"[CAMERA] Silent face recognition failed: {e}")

    if recognized_name:
        personalized_prompt = (
            f"The person in front of the camera is {recognized_name}. "
            f"{visual_question} "
            f"Give a natural 1-2 sentence response. If the answer is just a person's name, "
            f"greet them warmly and mention something you observe about them or the scene."
        )
    else:
        personalized_prompt = visual_question

    image_b64 = camera.numpy_frame_to_base64(frame_rgb, quality=75)

    if image_b64 is None:
        return "My eyes are fuzzy. Let me try again in a sec."

    from assistant.personalities import get_active_loader
    _identity = get_active_loader().get_reflection_hints().get("identity", "a desktop companion")

    if recognized_name:
        vision_system = (
            f"You are {config.ASSISTANT_NAME_DISPLAY}, {_identity}. "
            f"You can see {recognized_name} in front of the camera. "
            f"Answer their question naturally and conversationally in 1-3 sentences. "
            f"You may address them by name naturally if it fits. "
            f"Do not use markdown or bullet points. "
            f"Do not say 'I can see that' — just answer directly."
        )
    else:
        vision_system = (
            f"You are {config.ASSISTANT_NAME_DISPLAY}, {_identity}. "
            "Answer the visual question naturally and conversationally in 1-3 sentences. "
            "Do not use markdown or bullet points. "
            "Do not say 'I can see that' — just answer directly as if talking to a friend."
        )

    answer = (await llm_module.get_vision_response(
        image_base64=image_b64,
        prompt=personalized_prompt,
        system_prompt=vision_system,
    )).text

    if answer == "__LLM_UNAVAILABLE__":
        return "My eyes are fuzzy. Let me try again in a sec."

    return answer


async def handle_pending_forget_face(text: str) -> str | None:
    import assistant.actions as _act

    if _act.pending_forget_face.payload is None:
        return None

    from .. import faces
    name = text.strip().title()
    if not name:
        _act.pending_forget_face.touch()
        return "I didn't catch a name. Whose face should I forget?"

    _act.pending_forget_face.clear()
    removed = faces.forget_face(name)
    if removed:
        return f"Done, I've forgotten {name}'s face."
    else:
        return f"I don't have a saved face for {name}."


@tool_registry.decorator("meet_face")
async def handle_meet_face(params: dict, llm_response: str, bridge=None) -> str:
    from ..io import camera
    from .. import faces
    from .. import config as _config
    import face_recognition as fr

    name = params.get("name", "").strip().title()

    if not _config.CAMERA_ENABLED:
        return "My camera is currently disabled. Type /set camera_enabled true to turn it on."

    if bridge:
        await bridge.send_command("play_animation", name="thinking")

    if not name:
        return personality_say("face_need_name")

    INVALID_NAMES = {"me", "myself", "i", "this", "here", "us", "we"}
    if name.lower() in INVALID_NAMES:
        return (
            "I need your actual name, not 'me'! "
            f"Try saying '{config.ASSISTANT_NAME_LOWER} this is' followed by your real name."
        )

    frame = camera.capture_camera_frame_numpy(camera_index=_config.CAMERA_INDEX)
    if frame is None:
        return "I couldn't access the camera. Check that it's connected and permissions are enabled."

    locations = fr.face_locations(frame, number_of_times_to_upsample=2)

    if len(locations) == 0:
        return "I couldn't find a face in the frame. Make sure you're visible to the camera and try again."

    if len(locations) > 1:
        return "I can see more than one face. Please make sure only you are in front of the camera, then try again."

    encodings = fr.face_encodings(frame, locations)
    if not encodings:
        return "I found a face but couldn't compute its encoding. Please try again."

    result, count = faces.add_face(name, encodings[0], frame)

    if result == "added":
        return personality_say("face_learned", name=name, count=1)
    elif result == "updated":
        return personality_say("face_learned", name=name, count=count)
    elif result == "improved":
        return f"Updated! I replaced a low-quality angle with this one. Still 5 encodings saved for you, {name}."


@tool_registry.decorator("recognize_face")
async def handle_recognize_face(params: dict, llm_response: str, bridge=None) -> str:
    from ..io import camera
    from .. import faces
    from .. import llm as llm_module
    from .. import config as _config
    import face_recognition as fr

    if not _config.CAMERA_ENABLED:
        return "My camera is currently disabled. Type /set camera_enabled true to turn it on."

    if faces.face_count() == 0:
        return (
            "I don't have any saved faces yet. You can introduce yourself by "
            f"saying '{config.ASSISTANT_NAME_LOWER} this is' followed by your name."
        )

    if bridge:
        await bridge.send_command("play_animation", name="thinking")

    frame = camera.capture_camera_frame_numpy(camera_index=_config.CAMERA_INDEX)
    if frame is None:
        return "I couldn't access the camera. Check that it's connected and permissions are enabled."

    locations = fr.face_locations(frame, number_of_times_to_upsample=2)

    if not locations:
        try:
            image_b64 = camera.numpy_frame_to_base64(frame, quality=75)
            if not image_b64:
                return "My eyes are fuzzy. Let me try again in a sec."

            from assistant.personalities import get_active_loader
            _identity = get_active_loader().get_reflection_hints().get("identity", "a desktop companion")

            vision_prompt = "There is no face visible in this image. Describe what you actually see in the camera view in 1-2 natural sentences."
            vision_system = (
                f"You are {config.ASSISTANT_NAME_DISPLAY}, {_identity}. "
                "The user asked who is in front of the camera but there is no face visible. "
                "Describe what you actually see — objects, environment, scene. "
                "Be natural and conversational. Never say 'I can see that'. 1-2 sentences max."
            )

            answer = (await llm_module.get_vision_response(
                image_base64=image_b64,
                prompt=vision_prompt,
                system_prompt=vision_system,
            )).text

            if answer == "__LLM_UNAVAILABLE__":
                return "I couldn't find a face and the vision model is unavailable right now."

            return answer
        except Exception:
            return "I couldn't find a face and the vision model is unavailable right now."

    encodings = fr.face_encodings(frame, locations)
    matches = faces.find_faces(encodings, tolerance=_config.FACE_RECOGNITION_TOLERANCE)

    recognized = [m for m in matches if m is not None]
    unknown_count = matches.count(None)
    total = len(matches)

    if not recognized:
        try:
            image_b64 = camera.numpy_frame_to_base64(frame, quality=75)
            if not image_b64:
                return "My eyes are fuzzy. Let me try again in a sec."

            from assistant.personalities import get_active_loader
            _identity = get_active_loader().get_reflection_hints().get("identity", "a desktop companion")

            vision_prompt = "There is a face in this image but I don't recognize them from my saved faces. Who or what does this person look like? Describe them or identify them if they appear to be a well-known public figure. Be natural and conversational in 1-2 sentences."
            vision_system = (
                f"You are {config.ASSISTANT_NAME_DISPLAY}, {_identity}. "
                "You are telling your user what you see in the camera. "
                "The user asked who this person is. You do not have this person saved in your face database. "
                "Describe who you see naturally — if they appear to be a celebrity or well-known figure, say so. "
                "If they are unknown to you as well, say so briefly. Never say 'I can see that'. 1-2 sentences max."
            )

            answer = (await llm_module.get_vision_response(
                image_base64=image_b64,
                prompt=vision_prompt,
                system_prompt=vision_system,
            )).text

            if answer == "__LLM_UNAVAILABLE__":
                return "I don't recognize this person and the vision model is unavailable right now."

            return "I don't have them saved, but " + answer
        except Exception:
            return "I don't recognize this person and the vision model is unavailable right now."

    if total == 1:
        return personality_say("face_recognized", name=recognized[0])
    else:
        if unknown_count == 0:
            names = " and ".join(recognized)
            return f"I can see {names}!"
        else:
            names = " and ".join(recognized)
            unknown_str = "someone I don't recognize" if unknown_count == 1 else f"{unknown_count} people I don't recognize"
            return f"I can see {names} and {unknown_str}."


@tool_registry.decorator("forget_face")
async def handle_forget_face(params: dict, llm_response: str, bridge=None) -> str:
    import assistant.actions as _act
    from .. import faces

    name = params.get("name", "").strip().title()

    if not name:
        _act.pending_forget_face.set({"waiting": True})
        return "Whose face should I forget? Say a name."

    removed = faces.forget_face(name)
    if removed:
        return f"Done, I've forgotten {name}'s face."
    else:
        return f"I don't have a saved face for {name}."
