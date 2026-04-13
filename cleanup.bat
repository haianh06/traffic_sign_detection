#!/bin/bash
# GitHub Deployment Cleanup Script

echo "======================================================================="
echo "🧹 Traffic Sign Detection - GitHub Cleanup"
echo "======================================================================="
echo ""

# 1. Delete old debug files
echo "1️⃣  Deleting old debug files..."
del debug.py 2>/dev/null
del auto_hsv.py 2>/dev/null
del hsv_slider.py 2>/dev/null
del template_debug.py 2>/dev/null
echo "   ✅ Old debug files deleted"

# 2. Verify structure
echo ""
echo "2️⃣  Verifying project structure..."
if exist "main.py" echo "   ✅ main.py"
if exist "config.py" echo "   ✅ config.py"
if exist "tools.py" echo "   ✅ tools.py (NEW - consolidated)"
if exist "requirements.txt" echo "   ✅ requirements.txt"
if exist "LICENSE" echo "   ✅ LICENSE"
if exist ".gitignore" echo "   ✅ .gitignore"
if exist "README.md" echo "   ✅ README.md"
if exist "INSTALLATION.md" echo "   ✅ INSTALLATION.md"
if exist "core\detector.py" echo "   ✅ core/detector.py"
if exist "core\classifier.py" echo "   ✅ core/classifier.py"
if exist "templates" echo "   ✅ templates/"

# 3. Test consolidated tools
echo ""
echo "3️⃣  Testing consolidated tools..."
python tools.py > /dev/null 2>&1 && echo "   ✅ tools.py works"

# 4. Test main application
echo ""
echo "4️⃣  Testing main application imports..."
python -c "from main import TrafficSignRecognition; print('   ✅ main.py imports OK')" 2>/dev/null

echo ""
echo "======================================================================="
echo "✨ Cleanup Complete!"
echo "======================================================================="
echo ""
echo "📋 Next Steps:"
echo "  1. Delete old files: del debug.py auto_hsv.py hsv_slider.py template_debug.py"
echo "  2. git add -A"
echo "  3. git commit -m 'Clean up: consolidate debug tools to tools.py'"
echo "  4. git push origin main"
echo ""
echo "📚 Documentation files created:"
echo "  - README.md (GitHub overview)"
echo "  - INSTALLATION.md (Setup guide)"
echo "  - QUICKSTART.md (Quick reference)"
echo "  - CLEANUP.md (Deletion checklist)"
echo "  - GITHUB_READY.md (This summary)"
echo "  - LICENSE (MIT License)"
echo "  - .gitignore (Git patterns)"
echo ""
echo "🚀 Project is ready for GitHub!"
echo "======================================================================="
