Learning

- Role fo the config directory wiht `settings.yaml` is to keep **all the tunable paramets** in one place outside the code. We want to avoid hard-coded values like max_new_toekns=64

- In general, cof stores parameters, settings, and options used in teh code and things that change often versus stays the same

- the yaml confgi focuses on **what** values to use. It is a control panel where we can set what model to use, how big the output should be, avg runs, and whether to use CPU or GPU

- "Logging" comes after we can 1) load config 2) load pomrpts 3) run a simple model once (such that we have something to log)

- Logging does not happen during config becauselogging is useful when there is ACTIVITY TO TRAVK like "loaded config sucessufly, " model laoded in 2.3 secs'. In prodiuction code, you typically add logging as soon as your first “real” actions happen

- Use python3.11 and torch, the 3.12 version makes it incompatible.

- lots of issues dealing with the model selection in the setting.yaml and the model_loader.py. When I ran the model_loader.py, the code work but the output was trash and not understood. This is due to teh small model that was trained on 15mb, I need to stick witht eh small gpt and move forward because I went and rewrote the 2 files above and dont want to do taht in the future when i have to switch from the small model that worked to teh model that should work. 

- on macOS, you can hit native issues (memory/threads/tokenizers), which showed up as the bus error. So: tiny works ; larger models stress the environment and reveal those mac-specific quirks. that stubborn macOS native “bus error” again. Your code is fine; it’s the environment + backend

- had to change the reuqirements, settings.yaml, and the model loader.py often. like ALOT. There are alot of backend issues, had to rewrite requirements multiple times. 

- when to have smoke test per scipt vs logging? should do smoke test on each script to see if it works and then add logging  adn then add logging to eahc scropt where I want logs? What is industry practice?

- stuck on logging vs smoke test. Also, Shoudl smoke test be in a separate script to uncomplicate things? Issue I have now is logging in the core script AND in the smoek test? 

- Need to brush up on logging info, debug, and error














 csv ->
 app/config_loader ->
 app/prompt_laoder -> 
 app/model_loader ->
 app/runtime_setup ->
 app/inference_batch ->
 app/inference_stream ->
 app/metrics_logger ->
 benchmark_logger ->
 app/logger_config -> REStart HERE: STILL NEED TO DO 
 
 (put into 
 model_loader...DONE
 inference_batch..DONE 
 inference_stream...DONE
 benchmark_logger....) 