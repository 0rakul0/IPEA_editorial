
2026-04-22 16:41:24.161 Uncaught app execution
Traceback (most recent call last):
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\streamlit\runtime\scriptrunner\exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
             ^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 689, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\streamlit_app.py", line 741, in <module>
    result, logs = _run_review(
                   ^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\streamlit_app.py", line 613, in _run_review
    result = run_conversation(
             ^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\src\editorial_docx\graph_chat.py", line 718, in run_conversation
    return run_prepared_review(
           ^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\src\editorial_docx\graph_chat.py", line 595, in run_prepared_review
    for update in agent_apps[agent].stream(initial_state, stream_mode="updates"):
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langgraph\pregel\main.py", line 2759, in stream
    for _ in runner.tick(
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langgraph\pregel\_runner.py", line 167, in tick
    run_with_retry(
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langgraph\pregel\_retry.py", line 126, in run_with_retry
    return task.proc.invoke(task.input, config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langgraph\_internal\_runnable.py", line 656, in invoke
    input = context.run(step.invoke, input, config, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langgraph\_internal\_runnable.py", line 400, in invoke
    ret = self.func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\src\editorial_docx\pipeline\orchestrator.py", line 62, in run
    response = _invoke_with_model_fallback(prompt, payload, operation=f"agente {agent}")
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\src\editorial_docx\pipeline\runtime.py", line 353, in _invoke_with_model_fallback
    raise last_exc
  File "D:\Users\b13001972777\github\IPEA_editorial\src\editorial_docx\pipeline\runtime.py", line 344, in _invoke_with_model_fallback
    return _invoke_with_retry(runnable, payload, operation=operation)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\src\editorial_docx\pipeline\runtime.py", line 154, in _invoke_with_retry
    return runnable.invoke(payload)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langchain_core\runnables\base.py", line 3157, in invoke
    input_ = context.run(step.invoke, input_, config)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langchain_core\language_models\chat_models.py", line 455, in invoke
    self.generate_prompt(
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langchain_core\language_models\chat_models.py", line 1198, in generate_prompt
    return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langchain_core\language_models\chat_models.py", line 1005, in generate
    self._generate_with_cache(
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langchain_core\language_models\chat_models.py", line 1310, in _generate_with_cache
    result = self._generate(
             ^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langchain_openai\chat_models\base.py", line 1504, in _generate
    _handle_openai_api_error(e)
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langchain_openai\chat_models\base.py", line 1499, in _generate
    raw_response = self.client.with_raw_response.create(**payload)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\openai\_legacy_response.py", line 367, in wrapped
    return cast(LegacyAPIResponse[R], func(*args, **kwargs))
                                      ^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\openai\_utils\_utils.py", line 287, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\openai\resources\chat\completions\completions.py", line 1211, in create
    return self._post(
           ^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\openai\_base_client.py", line 1314, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\openai\_base_client.py", line 1087, in request
    raise self._make_status_error_from_response(err.response) from None
openai.NotFoundError: 404 page not found
During task with name 'gramatica_ortografia' and id '6ac2180c-b505-9ea3-6b58-6d2f5338ee1d'
2026-04-22 16:41:48.717 Uncaught app execution
Traceback (most recent call last):
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\streamlit\runtime\scriptrunner\exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
             ^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 689, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\streamlit_app.py", line 741, in <module>
    result, logs = _run_review(
                   ^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\streamlit_app.py", line 613, in _run_review
    result = run_conversation(
             ^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\src\editorial_docx\graph_chat.py", line 718, in run_conversation
    return run_prepared_review(
           ^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\src\editorial_docx\graph_chat.py", line 595, in run_prepared_review
    for update in agent_apps[agent].stream(initial_state, stream_mode="updates"):
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langgraph\pregel\main.py", line 2759, in stream
    for _ in runner.tick(
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langgraph\pregel\_runner.py", line 167, in tick
    run_with_retry(
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langgraph\pregel\_retry.py", line 126, in run_with_retry
    return task.proc.invoke(task.input, config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langgraph\_internal\_runnable.py", line 656, in invoke
    input = context.run(step.invoke, input, config, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langgraph\_internal\_runnable.py", line 400, in invoke
    ret = self.func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\src\editorial_docx\pipeline\orchestrator.py", line 62, in run
    response = _invoke_with_model_fallback(prompt, payload, operation=f"agente {agent}")
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\src\editorial_docx\pipeline\runtime.py", line 353, in _invoke_with_model_fallback
    raise last_exc
  File "D:\Users\b13001972777\github\IPEA_editorial\src\editorial_docx\pipeline\runtime.py", line 344, in _invoke_with_model_fallback
    return _invoke_with_retry(runnable, payload, operation=operation)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\src\editorial_docx\pipeline\runtime.py", line 154, in _invoke_with_retry
    return runnable.invoke(payload)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langchain_core\runnables\base.py", line 3157, in invoke
    input_ = context.run(step.invoke, input_, config)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langchain_core\language_models\chat_models.py", line 455, in invoke
    self.generate_prompt(
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langchain_core\language_models\chat_models.py", line 1198, in generate_prompt
    return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langchain_core\language_models\chat_models.py", line 1005, in generate
    self._generate_with_cache(
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langchain_core\language_models\chat_models.py", line 1310, in _generate_with_cache
    result = self._generate(
             ^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langchain_openai\chat_models\base.py", line 1504, in _generate
    _handle_openai_api_error(e)
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\langchain_openai\chat_models\base.py", line 1499, in _generate
    raw_response = self.client.with_raw_response.create(**payload)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\openai\_legacy_response.py", line 367, in wrapped
    return cast(LegacyAPIResponse[R], func(*args, **kwargs))
                                      ^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\openai\_utils\_utils.py", line 287, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\openai\resources\chat\completions\completions.py", line 1211, in create
    return self._post(
           ^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\openai\_base_client.py", line 1314, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Users\b13001972777\github\IPEA_editorial\.venv\Lib\site-packages\openai\_base_client.py", line 1087, in request
    raise self._make_status_error_from_response(err.response) from None
openai.NotFoundError: 404 page not found
During task with name 'gramatica_ortografia' and id 'cf5d0cd6-fb07-4627-bc18-3b9aebd5adc8'

