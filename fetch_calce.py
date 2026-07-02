import sys
import glob
import pandas as pd

def find_col(cols, *keywords):
    for c in cols:
        cl = c.lower().replace(" ", "").replace("_", "")
        if all(k in cl for k in keywords):
            return c
    return None

def main(in_dir, out_path):
    files = glob.glob(f"{in_dir}/*.xlsx") + glob.glob(f"{in_dir}/*.xls")
    if not files:
        print(f"No .xlsx/.xls files found in {in_dir}")
        sys.exit(1)

    frames = []
    for fp in files:
        try:
            xls = pd.ExcelFile(fp)
            # Arbin exports usually put the real data on a sheet containing "Channel" or "Record"
            sheet = next((s for s in xls.sheet_names if "channel" in s.lower()
                          or "record" in s.lower() or "sheet1" in s.lower()), xls.sheet_names[-1])
            df = xls.parse(sheet)
        except Exception as e:
            print(f"Skipping {fp}: {e}")
            continue

        cols = list(df.columns)
        time_c = find_col(cols, "test", "time") or find_col(cols, "time")
        cur_c = find_col(cols, "current")
        volt_c = find_col(cols, "voltage")
        temp_c = find_col(cols, "temp") or find_col(cols, "aux", "temp")

        if not (time_c and cur_c and volt_c):
            print(f"Could not identify required columns in {fp}. Columns found: {cols}")
            continue

        clean = pd.DataFrame({
            "Test_Time_s": df[time_c],
            "Current_A": df[cur_c],
            "Voltage_V": df[volt_c],
        })
        if temp_c:
            clean["Aux_Temperature_C"] = df[temp_c]
        clean["source_file"] = fp
        frames.append(clean)
        has_temp = temp_c is not None
        print(f"Parsed {fp}: {len(clean)} rows, temperature column {'FOUND: ' + temp_c if has_temp else 'NOT FOUND'}")

    if not frames:
        print("No files successfully parsed.")
        sys.exit(1)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows to {out_path}")
    print("\nSummary:")
    print(out.describe())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_calce.py <input_dir_of_xlsx_files> <output_csv_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])