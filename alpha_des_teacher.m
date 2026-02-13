
function a_des = alpha_des_teacher(a_star, a_prev, BETA, A_DOT_UP, A_DOT_DN, Ts)
% 反解 governor：给出应当下发的 a_des，使 governor 后落在 a_star
    epsl = 1e-12;
    if (1-BETA) < epsl
        a_lin = a_prev + A_DOT_UP*Ts;
    else
        a_lin = (a_star - BETA*a_prev) / (1 - BETA);
    end
    lo = a_prev - A_DOT_DN*Ts;
    hi = a_prev + A_DOT_UP*Ts;
    a_des = min(hi, max(lo, a_lin));
    a_des = min(1.0, max(0.0, a_des));
end
