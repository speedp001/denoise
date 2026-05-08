import struct
import numpy as np

from pathlib import Path


def read_ply_header(input_path):
    type_map = {
        "char": "b",
        "int8": "b",
        "uchar": "B",
        "uint8": "B",
        "short": "h",
        "int16": "h",
        "ushort": "H",
        "uint16": "H",
        "int": "i",
        "int32": "i",
        "uint": "I",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }
    
    int_types = {"char", "int8", "short", "int16", "int", "int32", "uchar", "uint8", "ushort", "uint16", "uint", "uint32"}
    float_types = {"float", "float32", "double", "float64"}

    with open(input_path, "rb") as f:
        first = f.readline().decode("ascii").strip()
        if first != "ply":
            raise ValueError("[Error] Not a valid PLY file:", input_path)

        fmt = None
        vertex_count = 0
        vertex_props = []
        in_vertex = False

        # Read PLY header
        while True:
            line = f.readline().decode("ascii").strip()
            if not line:
                continue
            if line == "end_header":
                break
            tokens = line.split()
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                in_vertex = tokens[1] == "vertex"
                if in_vertex:
                    vertex_count = int(tokens[2])
                    vertex_props = []
            elif tokens[0] == "property" and in_vertex:
                if len(tokens) != 3:
                    raise ValueError("[Error] List properties inside vertex are not supported:", line)
                vertex_props.append((tokens[1], tokens[2]))

        if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
            raise ValueError("[Error] Unsupported PLY format:", fmt)

        prop_names = [name for _, name in vertex_props]
        if not all(k in prop_names for k in ("x", "y", "z")):
            raise ValueError("[Error] Vertex properties must include x, y, z. Found:", prop_names)

        vertex_rows = []
        # ASCII format
        if fmt == "ascii":
            for _ in range(vertex_count):
                cols = f.readline().decode("ascii").strip().split()
                if len(cols) < len(vertex_props):
                    raise ValueError("[Error] Invalid ASCII PLY vertex row in:", input_path)
                row = []
                for (ptype, _), raw in zip(vertex_props, cols):
                    if ptype in int_types:
                        row.append(int(round(float(raw))))
                    elif ptype in float_types:
                        row.append(float(raw))
                    else:
                        raise ValueError("[Error] Unsupported PLY property type:", ptype)
                vertex_rows.append(row)
        # Binary format
        else:
            endian = "<" if fmt == "binary_little_endian" else ">"
            codes = []
            for ptype, _ in vertex_props:
                if ptype not in type_map:
                    raise ValueError("[Error] Unsupported PLY property type:", ptype)
                codes.append(type_map[ptype])
            row_struct = struct.Struct(endian + "".join(codes))
            for _ in range(vertex_count):
                raw = f.read(row_struct.size)
                if len(raw) != row_struct.size:
                    raise ValueError("[Error] Unexpected EOF while reading vertices from:", input_path)
                vals = row_struct.unpack(raw)
                row = []
                for (ptype, _), val in zip(vertex_props, vals):
                    if ptype in int_types:
                        row.append(int(round(float(val))))
                    elif ptype in float_types:
                        row.append(float(val))
                    else:
                        raise ValueError("[Error] Unsupported PLY property type:", ptype)
                vertex_rows.append(row)

    return np.array(vertex_rows, dtype=np.float64)


def convert_all_ply_in_data(data_root):
    ply_files = sorted(data_root.rglob("*.ply"))
    if not ply_files:
        print("[Error] No .ply files found:", data_root)
        return 0, 0, 0

    success_count = 0
    error_count = 0

    for ply_file in ply_files:
        npy_file = ply_file.with_suffix(".npy")
        try:
            points = read_ply_header(ply_file)
            np.save(npy_file, points)
            success_count += 1
            print("[OK]", ply_file, "->", npy_file)
        except Exception as exc:
            error_count += 1
            print("[ERROR]", ply_file, "reason:", str(exc))

    return success_count, error_count, len(ply_files)


if __name__ == "__main__":
    data_root = Path("./data")
    data_root.mkdir(parents=True, exist_ok=True)
    success_count, error_count, total_count = convert_all_ply_in_data(data_root)
    print("[DONE] success:", success_count, "error:", error_count, "total:", total_count)
