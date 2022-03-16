abstract type AbstractOutput end
abstract type no_save end
abstract type save_file end
abstract type save_reward end

mutable struct Output{S} <: AbstractOutput
    f_list::AbstractVector
    opt_buffer::AbstractVector
    res_file::AbstractVector
    io_buffer::AbstractVector
end

function set_f!(output::AbstractOutput, f::Number)
    append!(output.f_list, f)
end

function set_buffer!(output::AbstractOutput, buffer...)
    output.opt_buffer = [buffer...]
end

function set_io!(output::AbstractOutput, buffer...)
    output.io_buffer = [buffer...]
end

Output{T}(opt::AbstractOpt) where {T} = Output{T}([], result(opt), res_file(opt), [])
Output(opt::AbstractOpt, save_type::Bool) =
    save_type ? Output{save_file}(opt) : Output{no_save}(opt)

save_type(::Output{save_file}) = :save_file
save_type(::Output{no_save}) = :no_save

function SaveFile(output::Output{no_save})
    open("f.csv", "a") do f
        writedlm(f, output.f_list)
    end
    for (res, file) in zip(output.opt_buffer, output.res_file)
        open(file, "a") do g
            writedlm(g, res)
        end
    end
end

function SaveFile(output::Output{save_file}) end

function SaveCurrent(output::Output{save_file})
    open("f.csv", "w") do f
        writedlm(f, output.f_list[end])
    end
    for (res, file) in zip(output.opt_buffer, output.res_file)
        open(file, "w") do g
            writedlm(g, res)
        end
    end
end

function SaveCurrent(output::Output{no_save}) end

function SaveReward(output::Output{save_file}, reward::Number) ## TODO: reset file
    open("reward.csv", "w") do r
        writedlm(r, reward)
    end
end

function SaveReward(output::Output{no_save}, reward::Number) end

function SaveReward(rewards)
    open("reward.csv", "a") do r
        writedlm(r, rewards)
    end
end
