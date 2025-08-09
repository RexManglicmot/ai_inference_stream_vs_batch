Learning

- Role fo the config directory wiht `settings.yaml` is to keep **all the tunable paramets** in one place outside the code. We want to avoid hard-coded values like max_new_toekns=64

- In general, cof stores parameters, settings, and options used in teh code and things that change often versus stays the same

- the yaml confgi focuses on **what** values to use. It is a control panel where we can set what model to use, how big the output should be, avg runs, and whether to use CPU or GPU

- "Logging" comes after we can 1) load config 2) load pomrpts 3) run a simple model once (such that we have something to log)

- Logging does not happen during config becauselogging is useful when there is ACTIVITY TO TRAVK like "loaded config sucessufly, " model laoded in 2.3 secs'. In prodiuction code, you typically add logging as soon as your first “real” actions happen
















 csv ->
 app/config_loader ->
 app/prompt_laoder -> 