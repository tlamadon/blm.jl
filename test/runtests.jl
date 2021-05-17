# include("src/all.jl")
using blm
using Test
# using Statistics
using Random

@testset "testing model parameters" begin

    nl = 4
    nk = 6
    dm = blm.DistributionModel(nl,nk);

    @test all(size(dm.A0) .== [nk, nl])
    @test all( abs.(sum(dm.pnewf1,dims=2) .- 1) .< 1e-4 )

end 

@testset "testing likelihood" begin

    nn = 1_000_000
    nt = 10
    nl = 3
    nk = 4

    Random.seed!(1234);
    dm = blm.DistributionModel(nl,nk);
    data = blm.simulate(dm,nn,nt);

    qs,lk = blm.distributional_posteriors(dm, data)

    qs2 = zeros(nn,nl)
    lk2 = zeros(nn,nl)
    blm.distributional_posteriors!(lk2, qs2, dm, data)

    @test all( qs2 .≈ qs)
    @test all( lk2 .≈ lk)
end 

@testset "testing EM algorithm" begin
    # we simulate the model
    nn = 1_000_000
    nt = 10
    nl = 3
    nk = 4

    tol = 1e-4

    dm = blm.DistributionModel(nl,nk);
    dm.S0 *= .1 
    dm.Sm *= .1 
    dm.Ss *= .1 
    dm.Su *= .1 

    Random.seed!(1234);
    data = blm.simulate(dm,nn,nt);
    dm2 = copy(dm)

    blm.distributional_em!(dm2,data,1)
    
    @test  mean( (dm.Au[:] .- dm2.Au[:]).^2) > 1e-20
    @test  mean( (dm.Au[:] .- dm2.Au[:]).^2) < tol

    @test  mean( (dm.Su[:] .- dm2.Su[:]).^2) > 1e-20
    @test  mean( (dm.Su[:] .- dm2.Su[:]).^2) < tol

    @test  mean( (dm.A0[:] .- dm2.A0[:]).^2) > 1e-20
    @test  mean( (dm.A0[:] .- dm2.A0[:]).^2) < tol

    @test  mean( (dm.S0[:] .- dm2.S0[:]).^2) > 1e-20
    @test  mean( (dm.S0[:] .- dm2.S0[:]).^2) < tol

    @test  mean( (dm.Am[:] .- dm2.Am[:]).^2) > 1e-20
    @test  mean( (dm.Am[:] .- dm2.Am[:]).^2) < tol

    @test  mean( (dm.lambda0[:] .- dm2.lambda0[:]).^2) > 1e-20
    @test  mean( (dm.lambda0[:] .- dm2.lambda0[:]).^2) < tol

    @test  mean( (dm.lambdan[:] .- dm2.lambdan[:]).^2) > 1e-20
    @test  mean( (dm.lambdan[:] .- dm2.lambdan[:]).^2) < tol

    @test  mean( (dm.lambda1[:] .- dm2.lambda1[:]).^2) > 1e-20
    @test  mean( (dm.lambda1[:] .- dm2.lambda1[:]).^2) < tol

    @test  mean( (dm.pnewfn[:] .- dm2.pnewfn[:]).^2) > 1e-20
    @test  mean( (dm.pnewfn[:] .- dm2.pnewfn[:]).^2) < tol

    @test  mean( (dm.pnewf0[:] .- dm2.pnewf0[:]).^2) > 1e-20
    @test  mean( (dm.pnewf0[:] .- dm2.pnewf0[:]).^2) < tol

    @test  mean( (dm.pnewf1[:] .- dm2.pnewf1[:]).^2) > 1e-20
    @test  mean( (dm.pnewf1[:] .- dm2.pnewf1[:]).^2) < tol

    @test  mean( (dm.delta[:] .- dm2.delta[:]).^2) > 1e-20
    @test  mean( (dm.delta[:] .- dm2.delta[:]).^2) < tol

    @test  mean( (dm.As[:] .- dm2.As[:]).^2) > 1e-20
    @test  mean( (dm.As[:] .- dm2.As[:]).^2) < tol

    @test  mean( (dm.Ss[:] .- dm2.Ss[:]).^2) > 1e-20
    @test  mean( (dm.Ss[:] .- dm2.Ss[:]).^2) < tol
end


@testset "testing EM algorithm Event Study" begin
    # we simulate the model
    nn = 5_000_000
    nl = 3
    nk = 4

    tol = 1e-4

    dm = blm.DistributionModelES(nl,nk);
    dm.S1 *= .1 
    dm.S2 *= .1 

    Random.seed!(1234);
    data = blm.simulate(dm,nn);
    dm2 = copy(dm)

    blm.distributional_em!(dm2,data,1)
    
    @test  mean( (dm.A1[:] .- dm2.A1[:]).^2) > 1e-20
    @test  mean( (dm.A1[:] .- dm2.A1[:]).^2) < tol

    @test  mean( (dm.S1[:] .- dm2.S1[:]).^2) > 1e-20
    @test  mean( (dm.S1[:] .- dm2.S1[:]).^2) < tol

    @test  mean( (dm.A2[:] .- dm2.A2[:]).^2) > 1e-20
    @test  mean( (dm.A2[:] .- dm2.A2[:]).^2) < tol

    @test  mean( (dm.S2[:] .- dm2.S2[:]).^2) > 1e-20
    @test  mean( (dm.S2[:] .- dm2.S2[:]).^2) < tol

    @test  mean( (dm.pk1[:] .- dm2.pk1[:]).^2) > 1e-20
    @test  mean( (dm.pk1[:] .- dm2.pk1[:]).^2) < tol
end
