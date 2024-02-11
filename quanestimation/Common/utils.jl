import Pkg

"""
    is_loaded(mod::AbstractString)

Return `true` if `mod` is loaded.
"""
function is_loaded(mod::AbstractString)
    mods = Base.loaded_modules
    return any(m -> m.name == mod, keys(mods))
end

function module_version(mod::Module)
    toml = Base.parsed_toml(joinpath(dirname(dirname(Pkg.pathof(mod))), "Project.toml"))
    return toml["version"]
end

function pycall_version()
    if isdefined(@__MODULE__, :PyCall)
        return module_version(PyCall)
    end
    return nothing
end

function pythoncall_version()
    if isdefined(@__MODULE__, :PythonCall)
        return module_version(PythonCall)
    end
    return nothing
end


function parse_project()
    return parse_project(Pkg.project().path)
end

function parse_project(path::AbstractString)
    fullpath = nothing
    if ! endswith(path, "Project.toml")
        for f in ("Project.toml", "JuliaProject.toml")
            _fullpath = joinpath(path, f)
            if isfile(_fullpath)
                fullpath = _fullpath
                break
            end
            if isnothing(fullpath)
                error("Can't find project file in $path")
            end
        end
    else
        fullpath = path
    end
    return Base.parsed_toml(fullpath)
end


function  get_pycall_libpython()
    pycall_jl = Base.find_package("PyCall")
    if isnothing(pycall_jl)
        return (nothing, "not installed")
    end
    deps_jl = joinpath(dirname(dirname(Base.find_package("PyCall"))), "deps", "deps.jl")
    if ! isfile(deps_jl)
        return (nothing, "not built")
    end
    include(deps_jl) # not a great way to do this
    return (libpython, "ok")
end
