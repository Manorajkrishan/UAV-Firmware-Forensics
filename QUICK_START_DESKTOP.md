# 🚀 Quick Start - Desktop Application

## ⚡ Fast Setup (3 Steps)

### Step 1: Install Dependencies
```bash
cd desktop
npm install
```

### Step 2: Run the App
**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

**Or manually:**
```bash
npm start
```

### Step 3: Use It!
1. App window opens automatically
2. Backend starts automatically
3. Drag a file or click to upload
4. Click "Analyze Firmware"
5. View results!

## 🎯 That's It!

The desktop app will:
- ✅ Auto-start the Python backend
- ✅ Open a beautiful GUI window
- ✅ Be ready to analyze firmware files

## 📦 Building Installer

To create an installer for distribution:

```bash
cd desktop
npm run build
```

This creates:
- **Windows**: `.exe` installer in `dist-electron/`
- **Mac**: `.dmg` file in `dist-electron/`
- **Linux**: `.AppImage` in `dist-electron/`

## 🆘 Need Help?

See `DESKTOP_APP_COMPLETE_GUIDE.md` for detailed documentation.


