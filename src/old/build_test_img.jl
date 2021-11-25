using DataFrames, DataFramesMeta, CSV
using NPZ
using Random, Statistics
using DSP,FFTW, AbstractFFTs,Interpolations
using Wavelets, ContinuousWavelets
using ProgressMeter
using BenchmarkTools
using Profile
using Base.Threads
println(nthreads())

const sampling_rate = 2048.0
const dt = 1. / sampling_rate
const M_solor = 1.9884*10.0^30.0
const G = 6.67430*10.0^(-11.0)
const c =  299_792_458.0
const save_dim = 528
const BASE_DIR = "/home/yuri/kaggle/g2net-gravitational-wave-detection/"
const t_4096 = collect(1:4096) ./ 4096
const t_save_dim = collect(1:save_dim) ./ save_dim;

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

function select_detail(h::Vector, t0, t1)
    idx0 = max(floor(Int, t0*sampling_rate), 1)
    idx1 = floor(Int, t1*sampling_rate)
    t = (collect(idx0:idx1) .-idx0) ./sampling_rate .+ t0
    return t, h[idx0:idx1]
end

function select_detail(h::Matrix, t0, t1)
    idx0 = max(floor(Int, t0*sampling_rate), 1)
    idx1 = floor(Int, t1*sampling_rate)
    t = (collect(idx0:idx1) .-idx0) ./sampling_rate .+ t0
    return t, h[idx0:idx1,:]
end

function plot_detail(h, t0, t1)
    t, h_detail = select_detail(h, t0, t1)
    fig, ax = plt.subplots(figsize=(16*(t1-t0)/0.5,4))
    ax.plot(t, h_detail, lw=2)
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
    # SNR =vals[:SNR] #  Vector{Float32}(undef, 4096)
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
        # h_pow = mean(abs.(h[h .!=0.0]))
        # @. h = h * s_pow/h_pow
        matched_filter_core!(s, h̃, vals, @view SNR_arr_3[:,i])
    end
end


function forloop_main(f::String, stype::String, template_signal_fft_arr::Array{ComplexF32,2})
    rg ="[0-f]_[0-f]_[0-f]_$stype"
    m = match(Regex(rg), f)

    sig = npzread(f) |> x-> reshape(x, 3, div(length(x), 3)) |> transpose |> Array{Float32}
    etime = div(size(sig, 1), sampling_rate)
    sig_time = Array{Float32, 2}(undef, (4096, 3))
    s = Array{Float32, 1}(undef, 4096)
    SNR_arr_3 = Array{Float32,3}(undef, (save_dim, save_dim, 3))
    SNR_arr_3_img = Array{UInt8,3}(undef, (save_dim, save_dim, 3))
    for (sig_idx, start_time) in enumerate(0:2:etime-2)
        idx = 4096*(sig_idx-1)+1
        sig_time .= sig[idx:idx+4095, :]
        for site_idx in 1:3 # (["LIGO-H", "LIGO-L", "Virgo"])
            # @show(start_time)
            s .= sig_time[:,site_idx]
            # @show(size(s),size(template_signal_fft_arr))
            matched_filter_main!(s, template_signal_fft_arr, @view SNR_arr_3[:,:,site_idx])
        end

        fname = "$(m.match)_T$(Int(start_time)).npy"
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
        npzwrite(joinpath(save_dir, fname), SNR_arr_3_img)
    end
    return nothing
end

function SNR_main()
    Mc_min = 0.5
    Mc_max = 50.0

    Mc_arr = range(Mc_min, Mc_max, length=save_dim)
    template_signal_fft_arr =template_signal_arr_fft(Mc_arr)
    global save_dir = "/home/yuri/kaggle/g2net-gravitational-wave-detection/dataset/matched_filter_test/"

    # for stype in ["no_sig"]
    stype ="test_sig"
    println(stype)
    files = []
    for f in readdir("/home/yuri/kaggle/g2net-gravitational-wave-detection/dataset/test_equalized/", join=true)
        rg ="[0-f]_[0-f]_[0-f]_$(stype)"
        if match(Regex(rg), f) !== nothing # Or if occursin(r"*.\.pdf", f) == true
            push!(files, f)
        end
    end
    # files = files[1:16]
    p = Progress(length(files); showspeed=true)
    for (count,f) in collect(enumerate(files))
        # println("$count / $(length(files))")
        forloop_main(f,stype, template_signal_fft_arr)
        next!(p)
    end
   
end

SNR_main()
