# include("src/all.jl")
using blm
using Test
# using Statistics
using Random
using DataFrames
using Chain

@testset "testing basic" begin
    nl = 4
    nk = 6
    dm = blm.DistributionModel(nl,nk);
    @test 1 == 1
end

# @testset "testing model parameters" begin

#     nl = 4
#     nk = 6
#     dm = blm.DistributionModel(nl,nk);

#     @test all(size(dm.A0) .== [nk, nl])
#     @test all( abs.(sum(dm.pnewf1,dims=2) .- 1) .< 1e-4 )

# end

# @testset "testing likelihood" begin

#     nn = 1_000_000
#     nt = 10
#     nl = 3
#     nk = 4

#     Random.seed!(1234);
#     dm = blm.DistributionModel(nl,nk);
#     data = blm.simulate(dm,nn,nt);

#     qs,lk = blm.distributional_posteriors(dm, data)

#     qs2 = zeros(nn,nl)
#     lk2 = zeros(nn,nl)
#     blm.distributional_posteriors!(lk2, qs2, dm, data)

#     @test all( qs2 .≈ qs)
#     @test all( lk2 .≈ lk)
# end

# @testset "testing EM algorithm" begin
#     # we simulate the model
#     nn = 1_000_000
#     nt = 10
#     nl = 3
#     nk = 4

#     tol = 1e-4

#     dm = blm.DistributionModel(nl,nk);
#     dm.S0 *= .1
#     dm.Sm *= .1
#     dm.Ss *= .1
#     dm.Su *= .1

#     Random.seed!(1234);
#     data = blm.simulate(dm,nn,nt);
#     dm2 = copy(dm)

#     blm.distributional_em!(dm2,data,1)

#     @test  mean( (dm.Au[:] .- dm2.Au[:]).^2) > 1e-20
#     @test  mean( (dm.Au[:] .- dm2.Au[:]).^2) < tol

#     @test  mean( (dm.Su[:] .- dm2.Su[:]).^2) > 1e-20
#     @test  mean( (dm.Su[:] .- dm2.Su[:]).^2) < tol

#     @test  mean( (dm.A0[:] .- dm2.A0[:]).^2) > 1e-20
#     @test  mean( (dm.A0[:] .- dm2.A0[:]).^2) < tol

#     @test  mean( (dm.S0[:] .- dm2.S0[:]).^2) > 1e-20
#     @test  mean( (dm.S0[:] .- dm2.S0[:]).^2) < tol

#     @test  mean( (dm.Am[:] .- dm2.Am[:]).^2) > 1e-20
#     @test  mean( (dm.Am[:] .- dm2.Am[:]).^2) < tol

#     @test  mean( (dm.lambda0[:] .- dm2.lambda0[:]).^2) > 1e-20
#     @test  mean( (dm.lambda0[:] .- dm2.lambda0[:]).^2) < tol

#     @test  mean( (dm.lambdan[:] .- dm2.lambdan[:]).^2) > 1e-20
#     @test  mean( (dm.lambdan[:] .- dm2.lambdan[:]).^2) < tol

#     @test  mean( (dm.lambda1[:] .- dm2.lambda1[:]).^2) > 1e-20
#     @test  mean( (dm.lambda1[:] .- dm2.lambda1[:]).^2) < tol

#     @test  mean( (dm.pnewfn[:] .- dm2.pnewfn[:]).^2) > 1e-20
#     @test  mean( (dm.pnewfn[:] .- dm2.pnewfn[:]).^2) < tol

#     @test  mean( (dm.pnewf0[:] .- dm2.pnewf0[:]).^2) > 1e-20
#     @test  mean( (dm.pnewf0[:] .- dm2.pnewf0[:]).^2) < tol

#     @test  mean( (dm.pnewf1[:] .- dm2.pnewf1[:]).^2) > 1e-20
#     @test  mean( (dm.pnewf1[:] .- dm2.pnewf1[:]).^2) < tol

#     @test  mean( (dm.delta[:] .- dm2.delta[:]).^2) > 1e-20
#     @test  mean( (dm.delta[:] .- dm2.delta[:]).^2) < tol

#     @test  mean( (dm.As[:] .- dm2.As[:]).^2) > 1e-20
#     @test  mean( (dm.As[:] .- dm2.As[:]).^2) < tol

#     @test  mean( (dm.Ss[:] .- dm2.Ss[:]).^2) > 1e-20
#     @test  mean( (dm.Ss[:] .- dm2.Ss[:]).^2) < tol
# end


# @testset "testing EM algorithm Event Study" begin
#     # we simulate the model
#     nn = 5_000_000
#     nl = 3
#     nk = 4

#     tol = 1e-4

#     dm = blm.DistributionModelES(nl,nk);
#     dm.S1 *= .1
#     dm.S2 *= .1

#     Random.seed!(1234);
#     data = blm.simulate(dm,nn);
#     dm2 = copy(dm)

#     blm.distributional_em!(dm2,data,1)

#     @test  mean( (dm.A1[:] .- dm2.A1[:]).^2) > 1e-20
#     @test  mean( (dm.A1[:] .- dm2.A1[:]).^2) < tol

#     @test  mean( (dm.S1[:] .- dm2.S1[:]).^2) > 1e-20
#     @test  mean( (dm.S1[:] .- dm2.S1[:]).^2) < tol

#     @test  mean( (dm.A2[:] .- dm2.A2[:]).^2) > 1e-20
#     @test  mean( (dm.A2[:] .- dm2.A2[:]).^2) < tol

#     @test  mean( (dm.S2[:] .- dm2.S2[:]).^2) > 1e-20
#     @test  mean( (dm.S2[:] .- dm2.S2[:]).^2) < tol

#     @test  mean( (dm.pk1[:] .- dm2.pk1[:]).^2) > 1e-20
#     @test  mean( (dm.pk1[:] .- dm2.pk1[:]).^2) < tol
# end

@testset "testing clustering" begin
    # we simulate the model
    nn = 5_000_000
    nl = 3
    nk = 4

    tol = 1e-4

    dm = blm.DistributionModelES(nl, nk);
    dm.S1 *= .1;
    dm.S2 *= .1;

    Random.seed!(1234);
    sim_df = blm.simulate(dm, nn);

    # Create unstacked dataframe
    unstack_df_1 = DataFrame(f=sim_df.f1, w=sim_df.w1, g_true=sim_df.j1);
    unstack_df_2 = DataFrame(f=sim_df.f2, w=sim_df.w2, g_true=sim_df.j2);
    unstack_df = vcat(unstack_df_1, unstack_df_2);

    # Estimate groups
    groups = blm.group_firms(unstack_df.w, unstack_df.f; nq=10, ng=3, nstart=100)[1];

    # Insert estimated groups into dataframe
    groups_unstack = zeros(size(unstack_df)[1]);
    for i=1:size(groups_unstack)[1]
        groups_unstack[i] = groups[unstack_df.f[i]];
    end
    unstack_df[:, :g_estimated] = round.(Int, groups_unstack);

    # First true groups
    # Get mean income
    w_mean_true =
        @chain unstack_df begin
            groupby(:g_true)
            select(:w => mean)
            select(:w_mean)
        end;
    unstack_df[:, :w_mean_true] = w_mean_true.w_mean;
    # Sort by mean income
    sort!(unstack_df, :w_mean_true);
    # Reorder groups
    unstack_df[:, "g_true_new"] .= 0;
    true_g = unique(unstack_df.g_true);
    for i=1:size(true_g)[1]
        unstack_df[unstack_df.g_true .== true_g[i], :g_true_new] .= i;
    end

    # Second estimated groups
    # Get mean income
    w_mean_estimated =
        @chain unstack_df begin
            groupby(:g_estimated)
            select(:w => mean)
            select(:w_mean)
        end;
    unstack_df[:, :w_mean_estimated] = w_mean_estimated.w_mean;
    # Sort by mean income
    sort!(unstack_df, :w_mean_estimated);
    # Reorder groups
    unstack_df[:, "g_estimated_new"] .= 0;
    estimated_g = unique(unstack_df.g_estimated);
    for i=1:size(estimated_g)[1]
        unstack_df[unstack_df.g_estimated .== estimated_g[i], :g_estimated_new] .= i;
    end

    # Compute percent correct groups
    pct_correct = sum(unstack_df.g_true_new .== unstack_df.g_estimated_new) / size(unstack_df)[1];

    @test pct_correct >= 0.65
    @test pct_correct >= 0.75
    @test pct_correct >= 0.85
end
