# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/thunlp/TritonBench - Apache License 2.0
import os
import json
import argparse

def calculate(path_gen, path_ref):
    get_ms = lambda data: [item["ms"] for item in data]
    get_gbs = lambda data: [item["GB/s"] for item in data]
    get_tflops = lambda data: [item["TFLOPS"] for item in data]
    avg = lambda mss: round(sum(mss[0]) / sum(mss[1]), 4)

    data_gen = json.loads(open(path_gen, 'r', encoding='utf-8').read())
    data_ref = json.loads(open(path_ref, 'r', encoding='utf-8').read())
    assert len(data_gen) == len(data_ref), ""
    
    ms_ref, ms_gen = get_ms(data_ref), get_ms(data_gen)
    ms = avg((ms_ref, ms_gen))


    efficiency = max(round(max(get_gbs(data_gen)) * 100 / 2039, 4), round(max(get_tflops(data_gen)) * 100 / 312, 4))
    efficiency1 = max(round(max(get_gbs(data_ref)) * 100 / 2039, 4), round(max(get_tflops(data_ref)) * 100 / 312, 4))
    if efficiency >= 100 or ms >= 10:
        assert False, f"{path_gen.split('/')[-1]} test failed!"
    # if efficiency1 > efficiency:
    #     print(f"金标好啊好11111: {efficiency} < {efficiency1}")
    # else:
    #     print(f"生成棒棒棒！！！: {efficiency} > {efficiency1}")
    return ms, efficiency

def statis(gen_folder):
    avg = lambda listt: round(sum(listt) / len(listt), 2)
    files = [f for f in os.listdir(gen_folder) if f.endswith(".json")]
    spdups, effcys = [], []
    # print("===="*40)
    assert len(files) > 0, f"No json files found in {gen_folder}"
    for f in files:
        path_gen = os.path.join(gen_folder, f)
        path_ref = os.path.join(ref_folder, f)
        
        try:
            ms, efficiency = calculate(path_gen, path_ref)
            # print(f"{f}: {ms}")
            # print(f"{f}: {efficiency}\n")
            spdups.append(ms)
            effcys.append(efficiency)
            with open(os.path.join(gen_folder, f.replace(".json", "_perf_data.json")), 'w', encoding='utf-8') as f1:
                json.dump({f: {"ms": ms, "efficiency": efficiency}}, f1, indent=4)
        except:
            print(f"Error processing {f}, skipping...")
            continue            

    with open(os.path.join(gen_folder, "efficiency.json"), 'w', encoding='utf-8') as f:
        json.dump({"speed_up": spdups, "efficiency": effcys}, f, indent=4)
    # print(f"{gen_folder},{avg(spdups)},{avg(effcys)}")
    # print(spdups)
    # print(f"\n{gen_folder.split('/')[-1]}")
    # print(f"speed up: {avg(spdups)}")
    # print(f"efficiency: {avg(effcys)}")
    # print("===="*40)


def arg_parser():
    parser = argparse.ArgumentParser(description='Efficiency statistics')
    parser.add_argument('--gen_folder', type=str, required=True, help='The generated folder path')
    parser.add_argument('--ref_folder', type=str, required=True, help='The reference folder path')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    gen_folder = args.gen_folder
    ref_folder = args.ref_folder
    statis(gen_folder)