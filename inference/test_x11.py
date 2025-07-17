# test_x11.py
import Xlib.display
import sys
import os

def test_x11_connection():
    try:
        print("🔍 Testing X11 connection...")
        print(f"DISPLAY: {os.environ.get('DISPLAY', 'NOT SET')}")
        
        # Try to connect to display
        display = Xlib.display.Display()
        print("✅ Connected to display successfully")
        
        # Get screen info
        screen = display.screen()
        print(f"✅ Screen: {screen._data['width_in_pixels']}x{screen._data['height_in_pixels']}")
        
        # Test basic color allocation
        try:
            # Try to allocate a simple color
            color = display.alloc_color(0, 65535, 0)  # Green
            print(f"✅ Color allocation successful: {color}")
        except Exception as e:
            print(f"⚠️  Color allocation failed: {e}")
            print("   This is normal in some WSL setups")
        
        # Test window creation
        try:
            window = screen.root.create_window(
                100, 100, 200, 200, 2,
                screen.root_depth,
                Xlib.X.InputOutput,
                Xlib.X.CopyFromParent
            )
            print("✅ Window creation successful")
            window.destroy()
        except Exception as e:
            print(f"❌ Window creation failed: {e}")
        
        display.close()
        return True
        
    except Exception as e:
        print(f"❌ X11 connection failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_x11_connection():
        print("\n🎉 X11 is working! You can run the game.")
    else:
        print("\n❌ X11 is not working. Check your setup:")
        print("   1. Run: export DISPLAY=:0")
        print("   2. Run: xhost +local:")
        print("   3. Make sure you have an X11 server running")
