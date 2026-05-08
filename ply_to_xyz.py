import struct
import numpy as np

from pathlib import Path


# struct format characters for various PLY property types
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
            raise ValueError("[Error] Not a valid PLY file:", str(input_path))

        fmt = None
        vertex_count = 0
        vertex_props = []
        in_vertex = False

        # Read PLY header
        while True:
            line = f.readline().decode("ascii").strip()
            
            # Skip empty line in header
            if not line:
                continue
            # End of header
            if line == "end_header":
                break
            
            tokens = line.split()
            if tokens[0] == "format":
                fmt = tokens[1]
            
            elif tokens[0] == "element":
                # Check vertex element
                in_vertex = tokens[1] == "vertex"
                
                if in_vertex:
                    # vertex count
                    vertex_count = int(tokens[2])
                    # vertex properties(type, value)
                    vertex_props = []
                    
            elif tokens[0] == "property" and in_vertex:
                if len(tokens) != 3:
                    raise ValueError("[Error] List properties inside vertex are not supported: " + line)
                vertex_props.append((tokens[1], tokens[2]))

        if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
            raise ValueError("[Error] Unsupported PLY format: " + str(fmt))

        prop_value = [name for _, name in vertex_props]
        if "x" not in prop_value or "y" not in prop_value or "z" not in prop_value:
            raise ValueError("[Error] Vertex properties must include x, y, z.")

        x_idx = prop_value.index("x")
        y_idx = prop_value.index("y")
        z_idx = prop_value.index("z")

        # Read values
        xyz = np.empty((vertex_count, 3), dtype=np.float64)

        # PLY format: ASCII
        if fmt == "ascii":   
            """
            ASCII PLY reference example (generic, human-readable body):
            Header:
                ply
                format ascii 1.0
                element vertex 2
                property double x
                property double y
                property double z
                property uchar red
                property uchar green
                property uchar blue
                end_header
            Body lines:
                1.250000 -0.500000 2.750000 255 120 30
                0.000000 3.140000 -1.000000 10 200 240
            xyz output keeps only x y z in that order.
            """
            
            for i in range(vertex_count):
                cols = f.readline().decode("ascii").strip().split()
                if len(cols) < len(vertex_props):
                    raise ValueError("[Error] Invalid ASCII PLY vertex row in:", str(input_path))
                
                row = []
                for (ptype, _), raw in zip(vertex_props, cols):
                    # vertex_props: [(double, x), (double, y), (double, z), (uchar, red), (uchar, green), (uchar, blue)]
                    # cols: ["1.250000", "-0.500000", "2.750000", "255", "120", "30"]
                    if ptype in int_types:
                        row.append(int(round(float(raw))))
                    elif ptype in float_types:
                        row.append(float(raw))
                    else:
                        raise ValueError("[Error] Unsupported PLY property type:", str(ptype))
                    
                xyz[i, 0] = float(row[x_idx])
                xyz[i, 1] = float(row[y_idx])
                xyz[i, 2] = float(row[z_idx])
                
        # PLY format: Binary
        else:
            endian = "<" if fmt == "binary_little_endian" else ">"
            """
            Binary PLY example from this workspace (actual ./data files):
            Header:
                ply
                format binary_little_endian 1.0
                element vertex 180641
                property double x
                property double y
                property double z
                property uchar red
                property uchar green
                property uchar blue
                end_header
            First vertex body bytes (hex):
                68 72 34 e0 55 42 22 40 b3 b0 25 62 06 57 f9 bf
                c3 87 d9 56 3e 62 11 c0 0b 0b 0c
            Unpacked first vertex values:
                x=9.129561430360994, y=-1.5837463220478127,
                z=-4.3459409303922625, red=11, green=11, blue=12
            """
            
            codes = []
            # Change type to struct format character
            for ptype, _ in vertex_props:
                if ptype not in type_map:
                    raise ValueError("[Error] Unsupported PLY property type: " + ptype)
                codes.append(type_map[ptype])

            row_struct = struct.Struct(endian + "".join(codes))
            for i in range(vertex_count):
                raw = f.read(row_struct.size)
                if len(raw) != row_struct.size:
                    raise ValueError("[Error] Unexpected EOF while reading vertices from: " + str(input_path))
                row = row_struct.unpack(raw)
                if vertex_props[x_idx][0] in int_types:
                    x_val = int(round(float(row[x_idx])))
                elif vertex_props[x_idx][0] in float_types:
                    x_val = float(row[x_idx])
                else:
                    raise ValueError("[Error] Unsupported PLY property type:", str(vertex_props[x_idx][0]))

                if vertex_props[y_idx][0] in int_types:
                    y_val = int(round(float(row[y_idx])))
                elif vertex_props[y_idx][0] in float_types:
                    y_val = float(row[y_idx])
                else:
                    raise ValueError("[Error] Unsupported PLY property type:", str(vertex_props[y_idx][0]))

                if vertex_props[z_idx][0] in int_types:
                    z_val = int(round(float(row[z_idx])))
                elif vertex_props[z_idx][0] in float_types:
                    z_val = float(row[z_idx])
                else:
                    raise ValueError("[Error] Unsupported PLY property type:", str(vertex_props[z_idx][0]))

                xyz[i, 0] = float(x_val)
                xyz[i, 1] = float(y_val)
                xyz[i, 2] = float(z_val)

    return xyz


def convert_all_ply_in_data(data_root):
    ply_files = sorted(data_root.rglob("*.ply"))
    if not ply_files:
        print("[Error] No .ply files found:", data_root)
        return 0, 0, 0

    success_count = 0
    error_count = 0

    for ply_file in ply_files:
        xyz_file = ply_file.with_suffix(".xyz")
        
        # Success to read the PLY file and write XYZ output
        try:
            points = read_ply_header(ply_file)
            np.savetxt(xyz_file, points, fmt="%.8f")
            success_count += 1
            print("[OK]", ply_file, "->", xyz_file)
        
        # Failure to read the PLY file or write XYZ output
        except Exception as exc:
            error_count += 1
            print("[ERROR]", ply_file, "reason:", str(exc))

    return success_count, error_count, len(ply_files)


if __name__ == "__main__":
    data_root = Path("./data")

    data_root.mkdir(parents=True, exist_ok=True)

    success_count, error_count, total_count = convert_all_ply_in_data(data_root)
    print("[DONE] success:", success_count, "error:", error_count, "total:", total_count)