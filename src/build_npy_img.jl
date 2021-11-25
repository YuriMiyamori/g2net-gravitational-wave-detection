# -*- coding: utf-8 -*-
using DataFrames, DataFramesMeta, CSV
using NPZ
using Random, Statistics
using DSP,FFTW, AbstractFFTs,Interpolations
using Wavelets, ContinuousWavelets
using ProgressMeter
using BenchmarkTools
using Profile
using StatsBase

using PyPlot
plt.style.use("seaborn-whitegrid");
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 12
rcParams["figure.figsize"] = [16,16]
rcParams["figure.dpi"] = 220;

const sampling_rate = 2048.0
const dt = 1. / sampling_rate
const M_solor = 1.9884*10.0^30.0
const G = 6.67430*10.0^(-11.0)
const c =  299_792_458.0
const save_dim = 528
const BASE_DIR = "/home/yuri/kaggle/g2net-gravitational-wave-detection/"
const BACKUP_DIR = "/media/ssd_backup/g2net/dataset/02_image_npy"

const save_dim = 528
const t_4096 = collect(1:4096) ./ 4096
const t_save_dim = collect(1:save_dim) ./ save_dim;

using Base.Threads
println(nthreads())

function fade!(signal; fade_length=0.03)
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
    nothing
end

function Mc(m1, m2)
    # charp mass
    m1 = m1 * M_solor
    m2 = m2 * M_solor
    return (m1*m2)^(3/5) * (m1+m2)^(-1/5) /M_solor
end

function template_signal(Mc; total_length=4096, signal_time=0.5)
    """
    インスパイラル期のシグナルを作成
    """
    # @assert m1 <= m2
    signal_length = floor(Int, signal_time*sampling_rate)

    ω(t) = c^3/(5*G*Mc)*t
    h(t) = (ω(t))^(-1/4) * cos(-2.0*(ω(t))^(5/8))
    h_sig = Vector{Float32}(undef, total_length)

    t_min = 1.0 / 100.0 #
    for i in 1:signal_length
        t = (signal_length-i)*dt +t_min
        h_sig[i] = h(t)
    end
    @. h_sig[signal_length+1:end] = 0.0
    @views h_view = h_sig[1:signal_length]
    fade!(h_view,fade_length=0.04)
    return Array(h_sig)
end

function template_signal_arr_fft(Mc_arr; total_length=4096, signal_time=0.5)
    """
    インスパイラル期のシグナルを作成
    """
    h̃_arr = Array{ComplexF32, 2}(undef, (total_length, length(Mc_arr)))
    # @assert m1 <= m2
    signal_length = floor(Int, signal_time*sampling_rate)
    @showprogress for (i, Mc) in enumerate(Mc_arr)
        h_sig = template_signal(Mc*M_solor)
        h̃ = FFTW.fft(h_sig)
        h̃_arr[:,i] .= h̃
    end
    return h̃_arr
end

function matched_filter_core!(s̃::Vector{ComplexF32}, h̃::Vector{ComplexF32}, vals::Dict{Symbol, Vector{ComplexF32}},SNR::SubArray{Float32})
    # matched filterを計算する，
    @assert length(s̃) == length(h̃) == 4096
    h̃_conj = vals[:h̃_conj] # :Vector{ComplexF32}(undef, 4096)
    optimal =vals[:optimal] #  Vector{ComplexF32}(undef, 4096)
    tmp =vals[:tmp]
    h̃_conj .= conj.(h̃)
    # freq = FFTW.fftfreq(length(s̃), sampling_rate)
    df = 0.16662598649418273  #freq[2] - freq[1]
    @. optimal = s̃ *h̃_conj *df
    FFTW.ifft!(optimal)
    optimal .*= sampling_rate
    @. tmp = h̃ * h̃_conj
    sigmasq = sum(tmp)*df
    sigma = sqrt(abs(sigmasq))
    itp = interpolate((t_4096, ), abs.(optimal./sigma), Gridded(Linear()))
    @. SNR = itp(t_save_dim)
    nothing
end

function matched_filter_main!(s, template_signal_fft_arr::Array{ComplexF32,2}, SNR_arr_3::SubArray{Float32})
    s_pow = mean(abs.(s))
    s = ComplexF32.(s)
    FFTW.fft!(s)
    vals = Dict(
        :h̃_conj => Vector{ComplexF32}(undef, 4096),
        :optimal => Vector{ComplexF32}(undef, 4096),
        :tmp => Vector{ComplexF32}(undef, 4096)
    )
    for i in 1:size(template_signal_fft_arr,2) # 528
        h̃ = template_signal_fft_arr[:, i]
        matched_filter_core!(s, h̃, vals, @view SNR_arr_3[:,i])
    end
end

function forloop_main(f::String, template_signal_fft_arr::Array{ComplexF32,2}, train_test)
    save_path = joinpath(save_dir, "$(split(basename(f), '_')[1])_img.npy")
    backup_path = joinpath(BACKUP_DIR, train_test, basename(save_path))
    # println(backup_path)
    # すでにファイルが作られている場合は飛ばす
    if isfile(backup_path) 
        # println("$backup_path is exist")
        return nothing
    end
    
    sig = npzread(f) |> x-> reshape(x, 3, 4096) |> transpose |> Array{Float32}
    SNR_arr_3 = Array{Float32,3}(undef, (save_dim, save_dim, 3))
    SNR_arr_3_img = Array{UInt8,3}(undef, (save_dim, save_dim, 3))
    
    # isfile(save_path) && return nothing
    for site_idx in 1:3 # (["LIGO-H", "LIGO-L", "Virgo"])
        matched_filter_main!(sig[:, site_idx], template_signal_fft_arr, @view SNR_arr_3[:,:,site_idx])
    end

    for k in 1:3
        for j in 1:save_dim
            for i in 1:save_dim
                elem= SNR_arr_3[i,j,k]
                if elem > 200.0
                    elem = 200.0
                end
                SNR_arr_3_img[i,j,k] = round(UInt8, elem * 255.0/200.0)
            end
        end
    end
    # @show save_path
    # fig, ax = plt.subplots()
    # ax.imshow(SNR_arr_3_img[:,:, 1])
    
    npzwrite(save_path, SNR_arr_3_img)
    
    return nothing
end

function SNR_main(train_test)
    Mc_min = 0.5
    Mc_max = 50.0
    input_dir = joinpath(BASE_DIR, "dataset/01_equalized/$(train_test)")

    Mc_arr = range(Mc_min, Mc_max, length=save_dim)
    template_signal_fft_arr =template_signal_arr_fft(Mc_arr)
    global save_dir = joinpath(BASE_DIR, "dataset/02_image_npy/$(train_test)")

    # for stype in ["no_sig"]
    # for stype in ["in_sig", "no_sig"]
    files = readdir(input_dir, join=true)
    # files = files[1:4]
    p = Progress(length(files); showspeed=true)
    @threads for (count,f) in collect(enumerate(files))
        # println("$count / $(length(files))")
        forloop_main(f, template_signal_fft_arr, train_test)
        next!(p)
    end
end

function main()
    global train_test 
    for train_test in ["train", "test"]
        
        SNR_main(train_test)
    end
end
main()
