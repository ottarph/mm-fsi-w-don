from pathlib import Path
import subprocess as sp

def main():

    DATA_DIR_PATH = Path("../mm-fsi-w-don-DATA")

    Path("TMP_DATA_DIR").mkdir(parents=True, exist_ok=True)
    sp.run(["cp", "-a", str(DATA_DIR_PATH) + "/.", "TMP_DATA_DIR"], check=True)

    sp.run(["mv", "TMP_DATA_DIR/learnext_dataset/learnext_period_p1", "dataset"], check=True)

    Path("grav_test/data").mkdir(parents=True, exist_ok=True)
    sp.run(["mv", "TMP_DATA_DIR/grav_test/max_deformations.xdmf", "grav_test/data"], check=True)
    sp.run(["mv", "TMP_DATA_DIR/grav_test/max_deformations.h5", "grav_test/data"], check=True)

    Path("deeponet_extension/data").mkdir(parents=True, exist_ok=True)
    sp.run(["mv", "TMP_DATA_DIR/mesh", "deeponet_extension/data/mesh"], check=True)
    sp.run(["mv", "TMP_DATA_DIR/warmstart/state", "deeponet_extension/data/warmstart_state"], check=True)

    Path("deeponet_extension/models").mkdir(parents=True, exist_ok=True)
    sp.run(["mv", "TMP_DATA_DIR/best_run_model", "deeponet_extension/models/pretrained_best_run"], check=True)

    sp.run(["rm", "-r", "TMP_DATA_DIR"], check=True)

    return

if __name__ == "__main__":
    main()
