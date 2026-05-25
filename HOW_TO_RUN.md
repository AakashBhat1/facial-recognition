# How To Run This Project — Step By Step

 
You will end up running two things:
1. A **backend** (a small web server on your laptop — port 8000)
2. The **demo** (uses your webcam, talks to the backend)

You need **both running at the same time** in **two separate terminal windows**.

---

## 0. What you need before starting

- A laptop with **Windows 10 or 11** (this guide is Windows-first; macOS/Linux notes are at the bottom).
- A **webcam** (built-in or USB). Close Zoom / Teams / Chrome tabs that might be hogging it.
- About **5 GB of free disk space** (PyTorch and the face model are chunky).
- **Internet** on first run only — it downloads the face model (~16 MB) once and caches it.
- About **20–30 minutes** the first time. Most of that is `pip install` chewing through dependencies. Be patient.

---

## 1. Install Python 3.12

1. Go to https://www.python.org/downloads/release/python-3120/
2. Scroll down. Click **Windows installer (64-bit)**.
3. Run the downloaded `.exe`.
4. **VERY IMPORTANT** — on the first screen of the installer, tick the checkbox at the bottom that says **"Add python.exe to PATH"**. If you miss this, nothing below will work.
5. Click **Install Now**. Wait until it says "Setup was successful". Close.

**Sanity check:** Press `Windows + R`, type `powershell`, hit Enter. A blue window opens. Type:

```powershell
python --version
```

Press Enter. You should see something like `Python 3.12.x`. If you see `'python' is not recognized` → you forgot the PATH checkbox. Uninstall Python, reinstall, tick the checkbox.

---

## 2. Unzip the project

1. Find the `.zip` file your friend sent you.
2. Right-click → **Extract All...** → pick a simple place like `C:\demo` and click **Extract**.
3. You should now have a folder, for example: `C:\demo\facial recognition\`
4. Inside it you should see folders named `backend`, `demo`, files `Dockerfile`, `README.md`, `requirements`, etc.

**Do not** leave the folder inside OneDrive or with weird Unicode characters in the path. Plain `C:\demo\` is safest.

---

## 3. Open PowerShell **inside** the project folder

1. Open File Explorer and navigate to your unzipped folder (e.g. `C:\demo\facial recognition\`).
2. Hold **Shift** and **right-click** in any empty area inside that folder.
3. Click **Open PowerShell window here** (or "Open in Terminal" on Windows 11).
4. A blue window opens. The prompt at the bottom should already show your project path.

Type this to double-check you're in the right place:

```powershell
ls
```

You should see `backend`, `demo`, `Dockerfile`, `README.md`, etc. listed. If you don't, you're in the wrong folder — go back and try again.

---

## 4. Create a virtual environment (a private Python sandbox)

A "venv" keeps this project's libraries from polluting your system Python. **Always** activate it before running anything.

Type **exactly** this and press Enter:

```powershell
python -m venv .venv
```

Wait ~10 seconds. Nothing visible happens, but a new hidden folder `.venv\` is now in your project. Don't touch it.

---

## 5. Activate the virtual environment

Type:

```powershell
.\.venv\Scripts\Activate.ps1
```

If it works, your prompt will now have `(.venv)` at the start. Like:

```
(.venv) PS C:\demo\facial recognition>
```

### If you see a red error mentioning "execution policy" or "running scripts is disabled"

Windows blocks scripts by default. Run this **once**, then retry the activate command above:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

(This only relaxes the rule for this one PowerShell window. It does NOT change anything globally.)

**Important:** Every new PowerShell window starts WITHOUT the venv active. If you close the window and open a new one, you must `cd` back to the project folder and run the activate command again.

---

## 6. Upgrade pip (Python's package installer)

Old pip versions choke on modern packages. Upgrade it:

```powershell
python -m pip install --upgrade pip
```

Takes ~10 seconds. You'll see a "Successfully installed pip-X.Y.Z" message.

---

## 7. Install all the project's libraries

This is the big one. It downloads PyTorch (~700 MB), OpenCV, InsightFace, and friends.

```powershell
pip install -r backend\requirements.txt
```

**This takes 5 to 20 minutes** depending on your internet. You will see hundreds of lines scroll by. **Do not close the window.** Do not interrupt with Ctrl+C.

When it's done you should see lots of `Successfully installed ...` followed by your prompt coming back. If you see red `ERROR:` lines at the very end, scroll up and read the first error — it usually tells you what's missing.

### Common install errors

- **`Microsoft Visual C++ 14.0 or greater is required`** — install "Build Tools for Visual Studio 2022" from https://visualstudio.microsoft.com/downloads/ (pick the **C++ build tools** workload), then re-run the `pip install` command.
- **`SSL certificate verify failed`** — your work laptop has a corporate proxy. Try from a home network or run `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r backend\requirements.txt`.
- **Hangs forever on `torch`** — torch is genuinely 700 MB. Give it 10 minutes before assuming it's stuck.

---

## 8. Create the configuration file

The backend needs a `.env` file telling it some settings. Create it now. Copy the whole block below — including the `@"` line and the closing `"@` line — and paste it into PowerShell as one chunk, then press Enter:

```powershell
@"
DATABASE_URL=sqlite:///./data/smart_locker.db
SECRET_KEY=please_change_me_to_any_random_string_at_least_16_chars
USE_BUFFALO_MODEL=true
USE_CUSTOM_FACE_MODEL=false
FACE_MODEL_NAME=buffalo_s
SIMILARITY_THRESHOLD=0.45
ML_SCORING_ENABLED=false
ANTISPOOF_ENABLED=false
QUALITY_BLUR_THRESHOLD=40
MULTI_FRAME_MIN_REQUIRED=3
"@ | Out-File -FilePath backend\.env -Encoding utf8
```

Verify it worked:

```powershell
Get-Content backend\.env
```

You should see the settings printed back at you.

---

## 9. Start the backend (Terminal #1)

Still in the same PowerShell window with `(.venv)` showing, type:

```powershell
uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

After about 10–60 seconds (it downloads the face model on first run) you should see something like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
Smart Locker API v2.0 is running.
```

**Leave this window open.** Closing it stops the backend. If you do close it by accident, just re-activate the venv and run the `uvicorn` command again.

**Sanity check:** Open a web browser and go to http://127.0.0.1:8000/docs — you should see an interactive API page. Cool? Cool.

---

## 10. Open a SECOND PowerShell window (Terminal #2)

The backend is now hogging the first window. You need a second one for the demo.

1. Repeat the **Shift + right-click → Open PowerShell window here** trick from step 3, in the same project folder.
2. Activate the venv in this new window too:

```powershell
.\.venv\Scripts\Activate.ps1
```

(If you get the execution policy error again, run the `Set-ExecutionPolicy ... Bypass` command from step 5.)

You now have:
- **Terminal #1**: backend running, do not touch.
- **Terminal #2**: empty `(.venv)` prompt, ready for the demo.

---

## 11. Enroll your face (Terminal #2)

```powershell
python demo\live_demo.py enroll --name "YourName"
```

A webcam window pops up. **Look at the camera.** You'll see a green box around your face and a counter going `1/7, 2/7, ...`. It captures 7 frames over a few seconds, then closes the window and POSTs them to the backend.

Console output should end with something like `Enrolled: id=1, name=YourName`.

---

## 12. Recognize yourself (Terminal #2)

```powershell
python demo\live_demo.py recognize
```

Webcam opens again. **Blink once** during capture (this is the liveness check — it confirms you're a real person, not a photo). Window closes, and you should see a confidence score and either `MATCH` or `DENY` in the terminal.

---

## 13. Run the kiosk simulation (Terminal #2)

This is the fancy interactive UI — a grid of lockers you can click on:

```powershell
python demo\locker_simulation.py
```

A 1280x720 window opens showing 6 lockers. Click one, then choose **Sign Up** to bind your face to that locker, or **Login** if you already enrolled. Press **Q** to quit, **ESC** to go back.

---

## 14. When you're done

- In **Terminal #2** (or after closing the demo window) — just close the window or press Ctrl+C.
- In **Terminal #1** — press **Ctrl+C** to stop the backend.
- Next time, you only need to do steps 5 (activate venv), 9 (start backend), 10–13 (demo). Steps 1–8 are one-time.

---

## Troubleshooting

| What you see | What to do |
|---|---|
| `python is not recognized` | You skipped the "Add Python to PATH" checkbox in step 1. Reinstall Python with the checkbox ticked. |
| `running scripts is disabled` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then retry the activate command. |
| `ModuleNotFoundError: No module named 'fastapi'` (or similar) | Your venv is not active. Look for `(.venv)` in your prompt. If missing, run `.\.venv\Scripts\Activate.ps1`. |
| `ERROR: Cannot open webcam` | Another app is using the camera. Close Zoom, Teams, Chrome, OBS, etc. Then retry. |
| `Connection refused` or `Could not reach http://127.0.0.1:8000` | The backend is not running. Check Terminal #1 — start it again with the `uvicorn` command from step 9. |
| Face model download fails on first run | Your internet is flaky or behind a strict proxy. Retry on a different network. |
| `Address already in use` when starting uvicorn | Something else is already on port 8000. Either close it, or change the port: `uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8001` (then update the demo with `--api http://127.0.0.1:8001`). |
| Demo windows are tiny / black | Update your webcam driver. Try a different USB port. |

---

## Reset everything (start fresh)

If something is hopelessly broken:

```powershell
# Delete the database (re-created automatically next start)
Remove-Item backend\data\smart_locker.db

# Delete the demo's locker assignments
Remove-Item demo\locker_assignments.json

# Nuke the venv and start over
Remove-Item -Recurse -Force .venv
```

Then go back to step 4.

---

## macOS / Linux quick notes

Everything is the same except:

- Install Python via `brew install python@3.12` (macOS) or your distro's package manager (Linux). On Linux you may also need `sudo apt install python3.12-venv libgl1 libglib2.0-0`.
- Activate the venv with `source .venv/bin/activate` instead of `.\.venv\Scripts\Activate.ps1`.
- Use forward slashes: `pip install -r backend/requirements.txt`, `python demo/live_demo.py enroll --name "You"`.
- No execution-policy issue.
- For the `.env` file, just create it in any text editor with the same contents shown in step 8.

---

## Running with Docker (optional, advanced)

If you'd rather run the backend in a container:

```powershell
docker build -t smart-locker-backend .
docker run --rm -p 8000:8000 --env-file backend\.env smart-locker-backend
```

The demos still run on the host (webcam access from Docker on Windows is painful). See the comments at the top of `Dockerfile` for the full command with volume mounts.

---

That's it. If you hit something this guide doesn't cover, take a screenshot of the error and send it back. Good luck!
