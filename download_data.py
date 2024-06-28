from pathlib import Path
import subprocess as sp
import urllib.request
import shutil

def main():

    DATA_URL = "https://zenodo.org/records/12582079/files/mm-fsi-w-don-DATA.tar.gz?download=1"

    with urllib.request.urlopen(DATA_URL) as response:
        with open("TMP_DATA_DIR.tar.gz", 'wb') as f:
            shutil.copyfileobj(response, f)

    shutil.unpack_archive("TMP_DATA_DIR.tar.gz", "TMP_DATA_DIR")


    sp.run(["mv", "TMP_DATA_DIR/learnext_dataset/learnext_period_p1", "dataset"], check=True)

    Path("grav_test/data").mkdir(parents=True, exist_ok=True)
    sp.run(["mv", "TMP_DATA_DIR/grav_test/max_deformations.xdmf", "grav_test/data"], check=True)
    sp.run(["mv", "TMP_DATA_DIR/grav_test/max_deformations.h5", "grav_test/data"], check=True)

    Path("deeponet_extension/data").mkdir(parents=True, exist_ok=True)
    sp.run(["mv", "TMP_DATA_DIR/mesh", "deeponet_extension/data/mesh"], check=True)
    sp.run(["mv", "TMP_DATA_DIR/warmstart/state", "deeponet_extension/data/warmstart_state"], check=True)

    Path("deeponet_extension/models").mkdir(parents=True, exist_ok=True)
    sp.run(["mv", "TMP_DATA_DIR/best_run_model", "deeponet_extension/models/pretrained_best_run"], check=True)


    Path("TMP_DATA_DIR.tar.gz").unlink()
    shutil.rmtree("TMP_DATA_DIR")

    return

if __name__ == "__main__":
    main()
