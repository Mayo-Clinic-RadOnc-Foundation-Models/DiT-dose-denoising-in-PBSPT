# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py


from . import gaussian_diffusion_con as gdc

from .respace_con import  SpacedDiffusion_con, space_timesteps_con




def create_diffusion_con(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = gdc.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gdc.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gdc.LossType.RESCALED_MSE
    else:
        loss_type = gdc.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion_con(
        use_timesteps=space_timesteps_con(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gdc.ModelMeanType.EPSILON if not predict_xstart else gdc.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gdc.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gdc.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gdc.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )


