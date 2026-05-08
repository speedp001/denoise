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

    return {
        "fmt": fmt,
        "vertex_count": vertex_count,
        "vertex_props": vertex_props,
        "vertex_rows": vertex_rows,
    }

def find_reference_ply(npy_file, data_root):
    stem = npy_file.stem
    tokens = stem.split("_")
    candidates = []
    if len(tokens) >= 2:
        candidates.append(tokens[1])
    if "(" in stem:
        candidates.append(stem.split("(", 1)[0].rstrip("_"))
    candidates.append(stem)
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ref_path = data_root / f"{candidate}.ply"
        if ref_path.exists():
            return ref_path
    return None

def write_ply_with_updated_xyz(output_path, ply_data, updated_rows):
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
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = ply_data["fmt"]
    vertex_props = ply_data["vertex_props"]
    mode = "wb" if fmt in {"binary_little_endian", "binary_big_endian"} else "w"
    
    with open(output_path, mode) as f:
        header_lines = [
            "ply",
            "format " + fmt + " 1.0",
            "element vertex " + str(len(updated_rows)),
        ]
        
        for ptype, pname in vertex_props:
            header_lines.append("property " + ptype + " " + pname)
        header_lines.append("end_header")
        header_text = "\n".join(header_lines) + "\n"
        
        if mode == "w":
            f.write(header_text)
            
            for row in updated_rows:
                out_cols = []
                
                for value, (ptype, _) in zip(row, vertex_props):
                    
                    if ptype in float_types:
                        out_cols.append(f"{float(value):.8f}")
                    else:
                        out_cols.append(str(int(value)))
                
                f.write(" ".join(out_cols) + "\n")
        
        else:
            f.write(header_text.encode("ascii"))
            endian = "<" if fmt == "binary_little_endian" else ">"
            codes = []
            for ptype, _ in vertex_props:
                if ptype not in type_map:
                    raise ValueError("[Error] Unsupported PLY property type: " + ptype)
                codes.append(type_map[ptype])
            row_struct = struct.Struct(endian + "".join(codes))
            
            for row in updated_rows:
                casted = []
                
                for value, (ptype, _) in zip(row, vertex_props):
                    if ptype in int_types:
                        casted.append(int(round(float(value))))
                    elif ptype in float_types:
                        casted.append(float(value))
                    else:
                        raise ValueError("[Error] Unsupported PLY property type:", str(ptype))
                
                f.write(row_struct.pack(*casted))

def convert_npy_to_ply(npy_file, output_path, reference_path):
    npy_xyz = np.load(npy_file)
    if npy_xyz.shape[1] < 3:
        raise ValueError("[Error] NPY file must contain at least 3 columns:", npy_file)
    
    ply_data = read_ply_header(reference_path)
    int_types = {"char", "int8", "short", "int16", "int", "int32", "uchar", "uint8", "ushort", "uint16", "uint", "uint32"}
    float_types = {"float", "float32", "double", "float64"}
    prop_names = [name for _, name in ply_data["vertex_props"]]
    x_idx = prop_names.index("x")
    y_idx = prop_names.index("y")
    z_idx = prop_names.index("z")
    ref_rows = ply_data["vertex_rows"]
    
    if len(npy_xyz) != len(ref_rows):
        raise ValueError("[Error] NPY and reference PLY vertex count mismatch:", npy_file, reference_path)
    updated_rows = []
    
    for i, point in enumerate(npy_xyz):
        src_row = list(ref_rows[i])
        x_type = ply_data["vertex_props"][x_idx][0]
        y_type = ply_data["vertex_props"][y_idx][0]
        z_type = ply_data["vertex_props"][z_idx][0]
        
        if x_type in int_types:
            src_row[x_idx] = int(round(float(point[0])))
        elif x_type in float_types:
            src_row[x_idx] = float(point[0])
        else:
            raise ValueError("[Error] Unsupported PLY property type:", x_type)
        
        if y_type in int_types:
            src_row[y_idx] = int(round(float(point[1])))
        elif y_type in float_types:
            src_row[y_idx] = float(point[1])
        else:
            raise ValueError("[Error] Unsupported PLY property type:", y_type)
        
        if z_type in int_types:
            src_row[z_idx] = int(round(float(point[2])))
        elif z_type in float_types:
            src_row[z_idx] = float(point[2])
        else:
            raise ValueError("[Error] Unsupported PLY property type:", z_type)
        
        updated_rows.append(src_row)
    
    write_ply_with_updated_xyz(output_path, ply_data, updated_rows)

def convert_all_npy_in_result(result_root, data_root):
    npy_files = sorted(result_root.rglob("*.npy"))
    
    if not npy_files:
        print("[Error] No .npy files found:", result_root)
        return 0, 0, 0
    success_count = 0
    error_count = 0
    
    for npy_file in npy_files:
        ply_file = npy_file.with_suffix(".ply")
        reference_file = find_reference_ply(npy_file, data_root)
        
        if reference_file is None:
            error_count += 1
            print("[ERROR] reference ply not found for:", npy_file)
            continue
        if ply_file.exists():
            print("[OVERWRITE]", ply_file)
        try:
            convert_npy_to_ply(npy_file, ply_file, reference_file)
            success_count += 1
            print("[OK]", npy_file, "->", ply_file)
        except Exception as exc:
            error_count += 1
            print("[ERROR]", npy_file, "reason:", str(exc))
    return success_count, error_count, len(npy_files)

if __name__ == "__main__":
    result_root = Path("./result")
    
    if not result_root.exists():
        result_root.mkdir(parents=True, exist_ok=True)
    data_root = Path("./data")
    
    if not result_root.exists():
        raise SystemExit("[Error] Result path does not exist:", result_root)
    
    if not data_root.exists():
        raise SystemExit("[Error] Data path does not exist:", data_root)
    
    success_count, error_count, total_count = convert_all_npy_in_result(result_root, data_root)
    print("[DONE] success:", success_count, "error:", error_count, "total:", total_count)
