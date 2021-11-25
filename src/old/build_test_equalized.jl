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

function read_npy_data!(path, res, num)
    data = npzread(path) |> x-> reshape(x, 3, sampling_rate*2) |> transpose |> Array
    for i in 1:3
        data[:,i] = fade!(data[:,i])
        res[(num-1)*4096+1:num*4096,i] .= data[:,i]
    end
    # res = cat(res, data,dims=1)
end

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


function check_in_sig(fnames, df, target)
    res = String[]
    for fname in fnames
        if first(df[df[!, :fname] .== fname, :target]) == target
            push!(res, fname)
        end
    end
    res
end

const number_of_chunks = 16

train_label_df = CSV.read(joinpath(BASE_DIR, "training_labels.csv"), DataFrame)
train_label_df[:, :fname] = train_label_df[:, :id] .* ".npy"
const sampling_rate = 2048
const dt = 1 / sampling_rate

function main(train_test)
    out_path = joinpath(BASE_DIR, "dataset/train_equalized")
    if train_test == "test"
        out_path = joinpath(BASE_DIR, "dataset/test_equalized")
    end

    @show(length(readdir(joinpath(BASE_DIR, train_test), join=true)))
    dir_0 = joinpath(BASE_DIR, train_test)
    for f1 in readdir(dir_0, join=false)
        dir_1 = joinpath(dir_0, f1)
        @showprogress for f2 in readdir(dir_1, join=false)
            dir_2 = joinpath(dir_1, f2)
            for f3 in readdir(dir_2, join=false)
                dir_3 = joinpath(dir_2, f3)
                # for (target, fname_add) in zip([1, 0], ["in_sig.npy", "no_sig.npy"])
                    fname_add = "test_sig.npy"
                    file_names = readdir(dir_3, join=false)
                    @show(file_names)
                    return 
                    # file_names = check_in_sig(file_names, train_label_df, target)
                    file_paths = String[]
                    # @show(target, file_names)
                    for file_name in file_names
                        push!(file_paths, joinpath(dir_3, file_name))
                    end

                    data = Array{Float64, 2}(undef, 4096*length(file_paths), 3)
                    for (num, file_path) in enumerate(file_paths)
                        read_npy_data!(file_path, data, num)
                    end
                    ##  main
                    h_equalized_filtered_arr = Float32[]
                    h̃_equalized_filtered_arr = ComplexF32[]
                    for i in 1:3 ## LIGO LIGO Virgo
                        h = data[:, i]
                        global t = collect(1:length(h)) / sampling_rate
                        global frequencies = FFTW.rfftfreq(length(h), sampling_rate)
                        h̃ = dt * FFTW.rfft(h)
                        h_equalized, h̃_equalized= noise_detect(h, h̃, number_of_chunks, 0.5)
                        responsetype = Bandpass(35, 300, fs=sampling_rate)
                        h_equalized_filtered = filt(digitalfilter(responsetype, Butterworth(4)), h_equalized)
                        h̃_equalized_filtered = dt * FFTW.rfft(h_equalized_filtered)
                        append!(h_equalized_filtered_arr, Float32.(h_equalized_filtered))
                        append!(h̃_equalized_filtered_arr, ComplexF32.(h̃_equalized_filtered))
                    end
                    h_equalized_filtered_arr = reshape(h_equalized_filtered_arr, (Int(length(h_equalized_filtered_arr)/3), 3))
                    h̃_equalized_filtered_arr = reshape(h̃_equalized_filtered_arr, (Int(length(h̃_equalized_filtered_arr)/3), 3))
                    npzwrite(joinpath(out_path, "$(f1)_$(f2)_$(f3)_$(fname_add)"),
                        h_equalized_filtered_arr |>  transpose |> Array)
                end # f3
        end
    end
end


main("test")
