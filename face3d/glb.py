"""Minimal glTF 2.0 binary (GLB) writer and validator.

Only what this project needs: triangle meshes with positions, normals,
vertex colours and an optional base-colour texture.
"""

from __future__ import annotations

import json
import struct

import numpy as np

FLOAT = 5126
USHORT = 5123
UINT = 5125
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963

_GLB_MAGIC = 0x46546C67
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942


class GlbBuilder:
    def __init__(self, generator: str = "EventEcho face3d"):
        self.generator = generator
        self._bin = bytearray()
        self.buffer_views: list[dict] = []
        self.accessors: list[dict] = []
        self.materials: list[dict] = []
        self.meshes: list[dict] = []
        self.nodes: list[dict] = []
        self.images: list[dict] = []
        self.textures: list[dict] = []
        self.samplers: list[dict] = []
        self._scenes_nodes: list[int] = []

    # -- buffers ---------------------------------------------------------
    def _view(self, data: bytes, target: int | None = None) -> int:
        while len(self._bin) % 4:
            self._bin.append(0)
        offset = len(self._bin)
        self._bin.extend(data)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def _accessor(self, data: bytes, comp_type: int, type_str: str, count: int,
                  target: int | None = None, minmax: np.ndarray | None = None) -> int:
        view = self._view(data, target)
        accessor = {"bufferView": view, "componentType": comp_type,
                    "count": count, "type": type_str}
        if minmax is not None:
            accessor["min"] = [float(x) for x in minmax[0]]
            accessor["max"] = [float(x) for x in minmax[1]]
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    # -- helpers ---------------------------------------------------------
    def add_image_png(self, png_bytes: bytes) -> int:
        view = self._view(png_bytes)
        self.images.append({"bufferView": view, "mimeType": "image/png"})
        self.samplers.append({"magFilter": 9729, "minFilter": 9987,
                              "wrapS": 10497, "wrapT": 10497})
        self.textures.append({"sampler": len(self.samplers) - 1,
                              "source": len(self.images) - 1})
        return len(self.textures) - 1

    def add_material(self, name: str, base_color=(1.0, 1.0, 1.0, 1.0),
                     roughness: float = 0.62, metallic: float = 0.0,
                     use_vertex_color: bool = True, texture: int | None = None,
                     double_sided: bool = False) -> int:
        pbr = {
            "baseColorFactor": [float(c) for c in base_color],
            "metallicFactor": float(metallic),
            "roughnessFactor": float(roughness),
        }
        if texture is not None:
            pbr["baseColorTexture"] = {"index": texture}
            pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
        material = {"name": name, "pbrMetallicRoughness": pbr}
        if double_sided:
            material["doubleSided"] = True
        self.materials.append(material)
        return len(self.materials) - 1

    def add_mesh(self, name: str, positions: np.ndarray, normals: np.ndarray,
                 indices: np.ndarray, material: int,
                 colors: np.ndarray | None = None,
                 uvs: np.ndarray | None = None) -> int:
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        normals = np.ascontiguousarray(normals, dtype=np.float32)
        count = positions.shape[0]

        attributes = {
            "POSITION": self._accessor(
                positions.tobytes(), FLOAT, "VEC3", count, ARRAY_BUFFER,
                np.stack([positions.min(axis=0), positions.max(axis=0)]),
            )
        }
        if normals is not None:
            attributes["NORMAL"] = self._accessor(normals.tobytes(), FLOAT, "VEC3", count, ARRAY_BUFFER)
        if colors is not None:
            colors = np.ascontiguousarray(colors, dtype=np.float32)
            attributes["COLOR_0"] = self._accessor(colors.tobytes(), FLOAT, "VEC3", count, ARRAY_BUFFER)
        if uvs is not None:
            uvs = np.ascontiguousarray(uvs, dtype=np.float32)
            attributes["TEXCOORD_0"] = self._accessor(uvs.tobytes(), FLOAT, "VEC2", count, ARRAY_BUFFER)

        indices = np.ascontiguousarray(indices, dtype=np.uint32 if count > 65535 else np.uint16)
        comp = UINT if indices.dtype == np.uint32 else USHORT
        index_accessor = self._accessor(indices.tobytes(), comp, "SCALAR",
                                        int(indices.size), ELEMENT_ARRAY_BUFFER)

        self.meshes.append({
            "name": name,
            "primitives": [{"attributes": attributes, "indices": index_accessor,
                            "material": material, "mode": 4}],
        })
        return len(self.meshes) - 1

    def add_node(self, mesh: int, name: str | None = None,
                 translation=None, rotation=None) -> int:
        node = {"mesh": mesh}
        if name:
            node["name"] = name
        if translation is not None:
            node["translation"] = [float(v) for v in translation]
        if rotation is not None:
            node["rotation"] = [float(v) for v in rotation]
        self.nodes.append(node)
        self._scenes_nodes.append(len(self.nodes) - 1)
        return len(self.nodes) - 1

    # -- output ----------------------------------------------------------
    def to_bytes(self) -> bytes:
        while len(self._bin) % 4:
            self._bin.append(0)
        document = {
            "asset": {"version": "2.0", "generator": self.generator},
            "scene": 0,
            "scenes": [{"name": "face", "nodes": list(self._scenes_nodes)}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{"byteLength": len(self._bin)}],
        }
        if self.images:
            document["images"] = self.images
            document["textures"] = self.textures
            document["samplers"] = self.samplers

        json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
        while len(json_bytes) % 4:
            json_bytes += b" "

        bin_bytes = bytes(self._bin)
        total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
        out = bytearray()
        out += struct.pack("<III", _GLB_MAGIC, 2, total)
        out += struct.pack("<II", len(json_bytes), _JSON_CHUNK)
        out += json_bytes
        out += struct.pack("<II", len(bin_bytes), _BIN_CHUNK)
        out += bin_bytes
        return bytes(out)


def validate(glb: bytes) -> list[str]:
    """Structural checks a strict glTF validator would also make."""
    problems: list[str] = []
    info = inspect(glb)
    if info["bytes"] > 2_000_000:
        problems.append(f"file is {info['bytes']} bytes, over the 2,000,000 limit")
    document = json.loads(glb[20:20 + struct.unpack_from("<I", glb, 12)[0]].decode("utf-8"))
    for node in document.get("nodes", []):
        if "mesh" not in node:
            problems.append("node without a mesh")
    for mesh in document.get("meshes", []):
        for prim in mesh["primitives"]:
            pos = document["accessors"][prim["attributes"]["POSITION"]]
            if "min" not in pos or "max" not in pos:
                problems.append("POSITION accessor without min/max")
            if "indices" in prim:
                idx = document["accessors"][prim["indices"]]
                if idx["count"] % 3:
                    problems.append("index count is not a multiple of 3")
                view = document["bufferViews"][idx["bufferView"]]
                start = view.get("byteOffset", 0)
                width = 4 if idx["componentType"] == UINT else 2
                raw = np.frombuffer(
                    glb[20 + struct.unpack_from("<I", glb, 12)[0] + 8 + start:],
                    dtype=np.uint32 if width == 4 else np.uint16,
                    count=idx["count"],
                )
                if raw.size and int(raw.max()) >= pos["count"]:
                    problems.append("index out of range")
            if not np.isfinite(pos["min"]).all() or not np.isfinite(pos["max"]).all():
                problems.append("non-finite POSITION bounds")
    return problems


def inspect(glb: bytes) -> dict:
    """Parse a GLB back and report the structural facts a validator checks."""
    if len(glb) < 20:
        raise ValueError("file too short to be a GLB")
    magic, version, length = struct.unpack_from("<III", glb, 0)
    if magic != _GLB_MAGIC:
        raise ValueError("bad GLB magic")
    if version != 2:
        raise ValueError(f"GLB version {version}, expected 2")
    if length != len(glb):
        raise ValueError(f"declared length {length} != actual {len(glb)}")

    offset = 12
    document = None
    bin_length = 0
    first = True
    while offset < len(glb):
        chunk_len, chunk_type = struct.unpack_from("<II", glb, offset)
        body = glb[offset + 8: offset + 8 + chunk_len]
        if first:
            if chunk_type != _JSON_CHUNK:
                raise ValueError("first chunk is not JSON")
            document = json.loads(body.decode("utf-8"))
            first = False
        elif chunk_type == _BIN_CHUNK:
            bin_length = chunk_len
        offset += 8 + chunk_len
    if document is None:
        raise ValueError("no JSON chunk")
    if document.get("asset", {}).get("version") != "2.0":
        raise ValueError("asset.version is not 2.0")

    declared = document["buffers"][0]["byteLength"]
    if declared > bin_length:
        raise ValueError(f"buffer needs {declared} bytes, BIN chunk holds {bin_length}")

    triangles = 0
    vertices = 0
    for mesh in document.get("meshes", []):
        for primitive in mesh["primitives"]:
            pos = document["accessors"][primitive["attributes"]["POSITION"]]
            vertices += pos["count"]
            if "indices" in primitive:
                triangles += document["accessors"][primitive["indices"]]["count"] // 3
            else:
                triangles += pos["count"] // 3
    return {
        "bytes": len(glb),
        "version": document["asset"]["version"],
        "generator": document["asset"].get("generator"),
        "meshes": len(document.get("meshes", [])),
        "materials": len(document.get("materials", [])),
        "vertices": vertices,
        "triangles": triangles,
        "textures": len(document.get("textures", [])),
        "bin_bytes": bin_length,
        "json_bytes": len(json.dumps(document)),
    }
