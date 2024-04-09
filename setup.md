The repository is separated into the 2D and 3D model parts

To run the 3D training application:
- The data files should be moved to the folder `model3Delastic/data/data_files`. They should all have the following format `data_elasticity_3D_128_i.pt` with `i` an integer starting from 0. The file `data_elasticity_3D_128_0.pt` should at least be present in the folder.
- From the command line/terminal, go to `model3Delastic/` subfolder and activate a python compiler/ conda environment with pytorch library et une version de Python supérieure à 3 (peut-être pas nécessaire).
- (It might be necessary to create the executable. Run `chmod +x scripts/script1.sh`.)
- Then execute `./scripts/script1.sh`
- The outputs are automatically saved. The data_files should then be moved to another directory before zipping the folder `model3Delastic/data/`
