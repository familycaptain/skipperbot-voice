# Wake Word Models

Put custom local wake-word models here.

Preferred stable model name:

```text
home_voice/wake_words/hey-skipper.onnx
```

Timestamped training exports like `Hey_Skipper_20260509_215130.onnx` also work.
If `VOICE_OPENWAKEWORD_MODEL_PATHS` is not set, the wake service auto-selects the
newest `Hey_Skipper*.onnx` file in this folder.

The wake service defaults to OpenWakeWord/ONNX so the same model path can be
used on this Windows server and later on the Raspberry Pi, unless we choose a
Pi-specific runtime.
