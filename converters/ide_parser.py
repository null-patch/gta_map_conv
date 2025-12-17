import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class IDEObject:
    id: int
    model_name: str
    texture_dictionary: str
    draw_distance: float
    flags: int
    section_type: str = "objs"
    file_path: str = ""


class IDEParser:

    SECTION_OBJS = "objs"

    def __init__(self):
        self.objects: Dict[int, IDEObject] = {}
        self.current_section: Optional[str] = None

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a single IDE file."""
        if not os.path.exists(file_path):
            logger.error(f"IDE file not found: {file_path}")
            return {}

        logger.info(f"Parsing IDE file: {file_path}")
        self.objects.clear()
        self.current_section = None

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            self._parse_lines(lines, file_path)

            logger.info(f"Parsed {len(self.objects)} objects from {file_path}")
            return {
                "objects": self.objects,
                "file_path": file_path,
            }
        except Exception as e:
            logger.error(f"Error parsing IDE file {file_path}: {e}")
            return {}

    def _parse_lines(self, lines: List[str], file_path: str):
        for line_number, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue

            lower = line.lower().rstrip(":")

            if lower == self.SECTION_OBJS:
                self.current_section = self.SECTION_OBJS
                continue
            if lower == "end":
                self.current_section = None
                continue

            if self.current_section == self.SECTION_OBJS:
                self._parse_objs_line(line, line_number, file_path)

    def _parse_objs_line(self, line: str, line_number: int, file_path: str):

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            logger.warning(f"Invalid objs line in {file_path}:{line_number} -> {parts}")
            return
        try:
            obj_id = int(parts[0])
            model_name = parts[1]
            tex_dict = parts[2]
            draw_dist = float(parts[3])
            flags_str = parts[4]

            if flags_str.lower().startswith("0x"):
                flags = int(flags_str, 16)
            else:
                flags = int(flags_str)

            obj = IDEObject(
                id=obj_id,
                model_name=model_name,
                texture_dictionary=tex_dict,
                draw_distance=draw_dist,
                flags=flags,
                section_type="objs",
                file_path=file_path,
            )
            self.objects[obj_id] = obj
        except Exception as e:
            logger.warning(
                f"Error parsing objs line in {file_path}:{line_number}: {e}"
            )

    def get_object_by_id(self, obj_id: int) -> Optional[IDEObject]:
        return self.objects.get(obj_id)


class IDEManager:
    """Manager for handling multiple IDE files."""

    def __init__(self):
        self.parsers: Dict[str, IDEParser] = {}
        self.master_parser = IDEParser()
        self.file_paths: List[str] = []

    def parse_directory(self, directory_path: str, recursive: bool = True) -> bool:
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return False

        try:
            ide_files: List[str] = []
            if recursive:
                for root, _, files in os.walk(directory_path):
                    for file in files:
                        if file.lower().endswith(".ide"):
                            ide_files.append(os.path.join(root, file))
            else:
                for file in os.listdir(directory_path):
                    if file.lower().endswith(".ide"):
                        ide_files.append(os.path.join(directory_path, file))

            logger.info(f"Found {len(ide_files)} IDE files in {directory_path}")

            for ide_file in ide_files:
                self.parse_file(ide_file)

            logger.info(
                f"Total objects parsed: {len(self.master_parser.objects)}"
            )
            return True
        except Exception as e:
            logger.error(f"Error parsing IDE directory {directory_path}: {e}")
            return False

    def parse_file(self, file_path: str) -> bool:
        """Parse a single IDE file and merge its objects into the master parser."""
        if not os.path.exists(file_path):
            logger.error(f"IDE file not found: {file_path}")
            return False

        parser = IDEParser()
        result = parser.parse_file(file_path)
        if not result or not parser.objects:
            logger.warning(f"No objects parsed from IDE file: {file_path}")
            return False

        self.parsers[file_path] = parser
        # Merge objects into master
        self.master_parser.objects.update(parser.objects)
        self.file_paths.append(file_path)

        logger.debug(
            f"Successfully parsed {file_path} ({len(parser.objects)} objects)"
        )
        return True

    def get_all_objects(self) -> Dict[int, IDEObject]:
        return self.master_parser.objects.copy()

    def get_object_by_id(self, obj_id: int) -> Optional[IDEObject]:
        return self.master_parser.get_object_by_id(obj_id)

    def clear(self):
        self.parsers.clear()
        self.master_parser = IDEParser()
        self.file_paths.clear()


def parse_ide_file(file_path: str) -> Dict[str, Any]:
    parser = IDEParser()
    return parser.parse_file(file_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        logging.basicConfig(level=logging.INFO)
        parser = IDEParser()
        result = parser.parse_file(test_file)
        if result:
            print(f"Successfully parsed {test_file}")
            print(f"Objects: {len(parser.objects)}")
        else:
            print(f"Failed to parse {test_file}")
    else:
        print("Usage: python ide_parser.py <ide_file_path>")
