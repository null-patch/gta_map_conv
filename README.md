# GTA Map Converter for Blender 2.79

Convert GTA Games world map assets (`.img`, `.ide`, `.ipl`, `.dff`, `.txd`) into clean, editable **OBJ** files readable by Blender 2.79. Supports textures, materials, object placement, and optional optimizations.

---

## 🚀 Features

- ✅ Read `.IMG`, `.IPL`, `.IDE`, `.DFF`, `.TXD`
- ✅ Export Blender-compatible `.OBJ` + `.MTL`
- ✅ Preserve object placements & map hierarchy
- ✅ Convert textures with optional resizing
- ✅ Multi-threaded DFF / TXD extraction
- ✅ Clean GUI built with PyQt5 for Linux Mint

---

## 🖼️ Screenshot

> Sample UI launching on Linux:

![screenshot](screenshots/gui.png)

> Output preview in Blender 2.79:

![preview](screenshots/preview.png)

---

## 📝 Requirements

- Python 3.8+
- PyQt5
- Pillow
- psutil
- numpy
- colorama

---

## 📦 Install

```bash
git clone https://github.com/null-patch/gta_map_conv.git
cd gta_map_conv
chmod +x install.sh
./install.sh
```

---

## ▶️ Run

```bash
source venv/bin/activate
python3 main.py
```

or:

```bash
./run.sh
```

---

## 🗂️ Folder Setup

Place your GTA assets in directories like:

```
GTA_SA_map/   → contains GTA3.IMG, etc.
maps/         → contains .ipl, .ide
export/       → output destination
```

Make sure `.img`, `.ipl`, and `.ide` files exist before starting conversion.

---

## ⚙️ Configurable Options

- Scale factor (GTA to Blender units)
- Coordinate system (Z-up or Y-up)
- Texture format (PNG, JPEG, BMP, TGA)
- Texture quality and size
- Combine meshes by material
- Generate LOD groups
- Enable/disable temp file cleanup

---

## 📁 Output

Creates:

```
export/
├── gta_map_export.obj
├── gta_map_export.mtl
├── textures/
│   ├── texture1.png
│   ├── ...
```

---

## 🙏 Credits

- Developed by **@null-patch**
- Reverse-engineering credit: GTA modding community
- Original assets © Rockstar Games

---

## 📜 License

MIT License. Free to use, remix, and contribute.

---

## 🔗 Links

- [Blender 2.79 Download](https://www.blender.org/download/previous-versions/)
- [GitHub Repo](https://github.com/null-patch/gta_map_conv)
