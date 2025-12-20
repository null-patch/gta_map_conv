import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class IPLPlacement:
    """Minimal representation of an IPL 'inst' placement."""
    id: int
    model_name: str
    interior: int
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    lod: int = -1
    flags: int = 0
    parent_id: int = -1
    file_path: str = ""
    line_number: int = 0


class IPLParser:
    """
    Minimal IPL parser for GTA SA.
    Parses only the 'inst' section, which defines placed objects.
    """

    def __init__(self):
        self.placements: List[IPLPlacement] = []
        self.current_section: Optional[str] = None
        self.file_path: str = ""

    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse a single IPL file and return a list of placement dicts."""
        if not os.path.exists(file_path):
            logger.error(f"IPL file not found: {file_path}")
            return []

        logger.info(f"Parsing IPL file: {file_path}")
        self.placements.clear()
        self.current_section = None
        self.file_path = file_path

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            self._parse_lines(lines)
            logger.info(
                f"Parsed {len(self.placements)} placements from {file_path}"
            )
            # Return list of dicts for compatibility with ConversionPipeline
            return [asdict(p) for p in self.placements]
        except Exception as e:
            logger.error(f"Error parsing IPL file {file_path}: {e}")
            return []

    def _parse_lines(self, lines: List[str]):
        line_number = 0

        for raw in lines:
            line_number += 1
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue

            lower = line.lower().rstrip(":")

            # Section markers
            if lower == "inst":
                self.current_section = "inst"
                continue
            if lower == "end":
                self.current_section = None
                continue

            # Only care about inst section for object placements
            if self.current_section == "inst":
                self._parse_inst_line(line, line_number)

    def _parse_inst_line(self, line: str, line_number: int):
        """
        GTA SA inst line format (simplified):
        ID, ModelName, Interior, X, Y, Z, RX, RY, RZ, SX, SY, [SZ], [LOD], [Flags]
        """
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 10:
            logger.warning(f"Invalid inst line (too few parts): {parts}")
            return

        try:
            obj_id = int(parts[0])
            model_name = parts[1]
            interior = int(parts[2])

            pos_x = float(parts[3])
            pos_y = float(parts[4])
            pos_z = float(parts[5])

            rot_x = float(parts[6])
            rot_y = float(parts[7])
            rot_z = float(parts[8])

            scale_x = float(parts[9])
            scale_y = float(parts[10])
            scale_z = scale_y

            idx = 11
            if idx < len(parts):
                # optional SZ
                try:
                    scale_z = float(parts[idx])
                    idx += 1
                except ValueError:
                    # if not a float, treat it as LOD/Flags
                    pass

            lod = -1
            if idx < len(parts):
                try:
                    lod = int(parts[idx])
                    idx += 1
                except ValueError:
                    lod = -1

            flags = 0
            if idx < len(parts):
                try:
                    flags = int(parts[idx])
                    idx += 1
                except ValueError:
                    flags = 0

            placement = IPLPlacement(
                id=obj_id,
                model_name=model_name,
                interior=interior,
                position=(pos_x, pos_y, pos_z),
                rotation=(rot_x, rot_y, rot_z),
                scale=(scale_x, scale_y, scale_z),
                lod=lod,
                flags=flags,
                file_path=self.file_path,
                line_number=line_number,
            )
            self.placements.append(placement)

        except Exception as e:
            logger.warning(f"Error parsing inst line {line_number}: {e}")


class IPLManager:
    """Manager for handling multiple IPL files."""

    def __init__(self):
        self.parsers: Dict[str, IPLParser] = {}
        self.all_placements: List[Dict[str, Any]] = []
        self.file_paths: List[str] = []

    def parse_directory(self, directory_path: str, recursive: bool = True) -> bool:
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return False

        try:
            ipl_files: List[str] = []

            if recursive:
                for root, _, files in os.walk(directory_path):
                    for file in files:
                        if file.lower().endswith(".ipl"):
                            ipl_files.append(os.path.join(root, file))
            else:
                for file in os.listdir(directory_path):
                    if file.lower().endswith(".ipl"):
                        ipl_files.append(os.path.join(directory_path, file))

            logger.info(f"Found {len(ipl_files)} IPL files in {directory_path}")

            for ipl_file in ipl_files:
                self.parse_file(ipl_file)

            logger.info(f"Total placements parsed: {len(self.all_placements)}")
            return True
        except Exception as e:
            logger.error(f"Error parsing IPL directory {directory_path}: {e}")
            return False

    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse a single IPL file and merge placements into master list."""
        if not os.path.exists(file_path):
            logger.error(f"IPL file not found: {file_path}")
            return []

        parser = IPLParser()
        placements = parser.parse_file(file_path)

        if placements:
            self.parsers[file_path] = parser
            self.all_placements.extend(placements)
            self.file_paths.append(file_path)
            logger.debug(
                f"Successfully parsed IPL file: {file_path} "
                f"({len(placements)} placements)"
            )
        else:
            logger.warning(f"No placements parsed from IPL file: {file_path}")

        return placements

    def get_all_placements(self) -> List[Dict[str, Any]]:
        return list(self.all_placements)

    def clear(self):
        self.parsers.clear()
        self.all_placements.clear()
        self.file_paths.clear()


def parse_ipl_file(file_path: str) -> List[Dict[str, Any]]:
    parser = IPLParser()
    return parser.parse_file(file_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        logging.basicConfig(level=logging.INFO)
        parser = IPLParser()
        placements = parser.parse_file(test_file)
        if placements:
            print(f"Successfully parsed {test_file}")
            print(f"Placements: {len(placements)}")
            print("\nFirst 5 placements:")
            for p in placements[:5]:
                print(
                    f"  {p['id']}: {p['model_name']} at {p['position']} "
                    f"(interior {p['interior']})"
                )
        else:
            print(f"Failed to parse {test_file}")
    else:
        print("Usage: python ipl_parser.py <ipl_file_path>")