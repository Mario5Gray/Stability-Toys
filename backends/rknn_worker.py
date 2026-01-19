# rknn_worker.py
from .base import PipelineWorker, GenSpec
from rknnlcm import RKNN2Model, RKNN2LatentConsistencyPipeline

# -----------------------------
# Pipeline Worker
# -----------------------------
class RKNNPipelineWorker(PipelineWorker):
    """
    Owns ONE pipeline instance. Execute jobs sequentially on this worker.
    """

    def __init__(
        self,
        worker_id: int,
        paths: ModelPaths,
        scheduler_config: Dict,
        tokenizer: CLIPTokenizer,
        rknn_context_cfg: Optional[dict] = None,
        use_rknn_context_cfgs: bool = True,
    ):
        super().__init__(worker_id)
        self.worker_id = worker_id
        self.paths = paths
        self.scheduler_config = scheduler_config
        self.tokenizer = tokenizer
        self.rknn_context_cfg = rknn_context_cfg or {}
        self.use_rknn_context_cfgs = use_rknn_context_cfgs

        self.pipe = None
        self._init_pipeline()

    def _mk_model(self, model_path: str, *, data_format: str) -> RKNN2Model:
        if self.use_rknn_context_cfgs:
            return RKNN2Model(model_path, data_format=data_format, **self.rknn_context_cfg)
        return RKNN2Model(model_path, data_format=data_format)

    def _init_pipeline(self):
        scheduler = LCMScheduler.from_config(self.scheduler_config)
        self.pipe = RKNN2LatentConsistencyPipeline(
            text_encoder=self._mk_model(self.paths.text_encoder, data_format="nchw"),
            unet=self._mk_model(self.paths.unet, data_format="nhwc"),
            vae_decoder=self._mk_model(self.paths.vae_decoder, data_format="nhwc"),
            scheduler=scheduler,
            tokenizer=self.tokenizer,
        )

    def run_job(self, job: Job) -> Tuple[bytes, int]:
        width, height = parse_size(job.req.size)
        seed = job.req.seed if job.req.seed is not None else gen_seed_8_digits()
        rng = np.random.RandomState(seed)

        result = self.pipe(
            prompt=job.req.prompt,
            height=height,
            width=width,
            num_inference_steps=job.req.num_inference_steps,
            guidance_scale=job.req.guidance_scale,
            generator=rng,
        )

        pil_image = result["images"][0]
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue(), seed
