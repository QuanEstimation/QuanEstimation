###################### mintime opt #############################
function mintime(::Val{:binary}, f, system)
    (; dynamics, output, obj) = system
    (; tspan, ctrl) = deepcopy(dynamics.data)
    low, high = 1, length(tspan)
    mid = 0
    f_mid = 0.0

    while low < high 
        mid = fld1(low + high, 2)

        dynamics.data.tspan = tspan[1:mid]
        dynamics.data.ctrl = [c[1:mid-1] for c in ctrl] 
        
        f_ini = objective(obj, dynamics)[2]

        if f > f_ini
            run(system)
            f_mid = output.f_list[end]
        else
            f_mid = f_ini
        end

        if abs(f-f_mid) < obj.eps
            break
        elseif f_mid < f
            low = mid + 1
        else
            high = mid - 1
        end
    end
    open("mtspan.csv","w") do t
        writedlm(t, dynamics.data.tspan)
    end
    open("controls.csv","w") do c
        writedlm(c, dynamics.data.ctrl)
    end
    println("The minimum time to reach target is ", dynamics.data.tspan[end],", data saved.")
end

function mintime(::Val{:forward}, f, system)
    (; dynamics, output, obj) = system
    (; tspan, ctrl) = deepcopy(dynamics.data)
    idx = 2
    f_now = 0.0

    while f_now < f && idx<length(tspan)
        dynamics.data.tspan = tspan[1:idx]
        dynamics.data.ctrl = [c[1:idx-1] for c in ctrl] 
        run(system)
        f_now = output.f_list[end]
        idx += 1
    end
    open("mtspan.csv","w") do t
        writedlm(t, dynamics.data.tspan)
    end
    open("controls.csv","w") do c
        writedlm(c, dynamics.data.ctrl)
    end
    println("The minimum time to reach target is ",dynamics.data.tspan[end],", data saved.")
end

mintime(s::String, args...) = mintime(Val{Symbol(s)}(), args...)
