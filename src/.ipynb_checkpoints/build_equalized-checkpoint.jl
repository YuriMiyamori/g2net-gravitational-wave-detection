# -*- coding: utf-8 -*-
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

const BASE_DIR = "/home/yuri/kaggle/g2net-gravitational-wave-detection/dataset"

const max_safe_exponent = 700.0

function fade!(signal, fade_length=0.06)
    n = length(signal)
    n0 = floor(Int, length(signal) * fade_length)
    n1 = n - n0
    t = collect(1:n)

    sigmoid(x)=1 ./(1 .+exp(-14 .*(x .-0.5)))
    for i in 1:n0
        signal[i] = signal[i] * sigmoid(i/n0)
    end
    for i in n1:n
        signal[i] = signal[i] * sigmoid((n-i)/(n-n1))
    end
    signal
end

# +
function read_npy_data!(path, res, num)
    data = npzread(path) |> x-> reshape(x, 3, sampling_rate*2) |> transpose |> Array
    for i in 1:3
        data[:,i] = fade!(data[:,i])
        res[(num-1)*4096+1:num*4096,i] .= data[:,i]
    end
    # res = cat(res, data,dims=1)
end

function read_single_npy_data(path, res, num)
    data = npzread(path) |> x-> reshape(x, 3, sampling_rate*2) |> transpose |> Array
    return data
end
# -

function noise_detect(h, h̃, number_of_chunks, overlap_rate)
    points_per_chunk = 2^floor(Int, log2(length(h)/number_of_chunks))

    s = welch_pgram(h, points_per_chunk, floor(Int, points_per_chunk*overlap_rate), fs=sampling_rate, window=hanning)
    f_noise, noise_welch = s.freq, s.power

    noise_spectral_density = sqrt.(2 * length(h) .* noise_welch ./ sampling_rate)
    nodes = (f_noise,)
    itp = interpolate(nodes, noise_spectral_density, Gridded(Linear()))
    noise_spectral_density_interpolator = itp(frequencies);

    h̃_equalized = h̃ ./ noise_spectral_density_interpolator
    h_equalized = (sampling_rate * FFTW.irfft(h̃_equalized, length(t)));

    return h_equalized, h̃_equalized
end


const number_of_chunks = 16

const sampling_rate = 2048
const dt = 1 / sampling_rate

function main(train_test)
    out_path = joinpath(BASE_DIR, "01_equalized/train")
    if train_test == "test"
        out_path = joinpath(BASE_DIR, "01_equalized/test")
    end

    @show(length(readdir(joinpath(BASE_DIR, train_test), join=true)))
    dir_0 = joinpath(BASE_DIR, train_test)
    for f1 in readdir(dir_0, join=false)
        dir_1 = joinpath(dir_0, f1)
        @showprogress for f2 in readdir(dir_1, join=false)
            dir_2 = joinpath(dir_1, f2)
            for f3 in readdir(dir_2, join=false)
                dir_3 = joinpath(dir_2, f3)
                file_names = readdir(dir_3, join=false)
                file_size = length(file_names)
                file_paths = String[]
                for file_name in file_names
                    push!(file_paths, joinpath(dir_3, file_name))
                end

                data = Array{Float64, 2}(undef, 4096*length(file_paths), 3)
                for (num, file_path) in enumerate(file_paths)
                    read_npy_data!(file_path, total_data, num)
                end
                ##  main
                h_equalized_filtered_arr = Float32[]
                for i in 1:3 ## LIGO LIGO Virgo
                    h = data[:, i]
                    global t = collect(1:length(h)) / sampling_rate
                    global frequencies = FFTW.rfftfreq(length(h), sampling_rate)
                    h̃ = dt * FFTW.rfft(h)
                    h_equalized, h̃_equalized= noise_detect(h, h̃, number_of_chunks, 0.5)
                    responsetype = Bandpass(35, 300, fs=sampling_rate)
                    h_equalized_filtered = filt(digitalfilter(responsetype, Butterworth(4)), h_equalized)
                    append!(h_equalized_filtered_arr, Float32.(h_equalized_filtered))
                end
                h_equalized_filtered_arr = reshape(h_equalized_filtered_arr, (Int(length(h_equalized_filtered_arr)/3), 3))
                
                for (i, file_name) in enumerate(file_names)
                    s = 4096*(i-1)+1
                    t = 4096*(i)
                    save_arr = h_equalized_filtered_arr[s:t, :]
                    save_name = split(file_name, '.')[1] * "equalized.npy"
                    npzwrite(joinpath(out_path, save_name),save_arr |>  transpose |> Array)
                end
            end
        end
    end
end

main("train")
