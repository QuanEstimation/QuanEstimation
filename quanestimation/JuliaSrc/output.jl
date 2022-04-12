abstract type AbstractOutput end
abstract type no_save end
abstract type savefile end
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

Output{T}(opt::AbstractOpt) where {T} = Output{T}([], [], res_file(opt), [])
Output(opt::AbstractOpt; save::Bool=false) =
    save ? Output{savefile}(opt) : Output{no_save}(opt)

save_type(::Output{savefile}) = :savefile
save_type(::Output{no_save}) = :no_save

function SaveFile(output::Output{no_save})
    open("f.csv", "w") do f
        writedlm(f, output.f_list)
    end
    for (res, file) in zip(output.opt_buffer, output.res_file)
        open(file, "w") do g
            writedlm(g, res)
        end
    end
end

function SaveFile(output::Output{savefile}) end

function SaveCurrent(output::Output{savefile})
    open("f.csv", "a") do f
        writedlm(f, output.f_list[end])
    end
    for (res, file) in zip(output.opt_buffer, output.res_file)
        open(file, "a") do g
            writedlm(g, res)
        end
    end
end

function SaveCurrent(output::Output{no_save}) end

function SaveReward(output::Output{savefile}, reward::Number) ## TODO: reset file
    open("reward.csv", "a") do r
        writedlm(r, reward)
    end
end

function SaveReward(output::Output{no_save}, reward::Number) end

function SaveReward(rewards)
    open("reward.csv", "w") do r
        writedlm(r, rewards)
    end
end
