using DataFrames, DataFramesMeta, CSV
using NPZ
using Random, Statistics
using DSP,FFTW, AbstractFFTs,Interpolations
using Wavelets, ContinuousWavelets
using ProgressMeter
using BenchmarkTools

using PyPlot
plt.style.use("seaborn-whitegrid");
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 12
rcParams["figure.figsize"] = [16,16]
rcParams["figure.dpi"] = 220;

const BASE_DIR = "/home/yuri/kaggle/g2net-gravitational-wave-detection/"


# B
# main("test")
#
name_df = DataFrame(dir1 = [], dir2=[], dir3=[], file_name=[], file_path=[], time=[], id=[])
function main(train_test)
    out_path = joinpath(BASE_DIR, "dataset/train_equalized")
    if train_test == "test"
        out_path = joinpath(BASE_DIR, "dataset/test_equalized")
    end

    # @show(length(readdir(joinpath(BASE_DIR, train_test), join=true)))
    dir_0 = joinpath(BASE_DIR, train_test)
    @showprogress for f1 in readdir(dir_0, join=false)
        dir_1 = joinpath(dir_0, f1)
         for f2 in readdir(dir_1, join=false)
            dir_2 = joinpath(dir_1, f2)
            for f3 in readdir(dir_2, join=false)
                dir_3 = joinpath(dir_2, f3)
                # for (target, fname_add) in zip([1, 0], ["in_sig.npy", "no_sig.npy"])
                fname_add = "test_sig.npy"
                file_names = readdir(dir_3, join=false)
                # file_names = check_in_sig(file_names, train_label_df, target)
                file_paths = String[]
                # @show(target, file_names)
                for file_name in file_names
                    push!(file_paths, joinpath(dir_3, file_name))
                end
                for (i, time) in enumerate(0:length(file_names)-1)
                    id = split(file_names[i], '.')[1]
                    push!(name_df, (dir_1, dir_2, dir_3, file_names[i], file_paths[i], Int(time*2), id))
                end
            end # f3
        end
    end
end

main("test")
CSV.write("file_name_df.csv", name_df)
@show(name_df)
