"""Machine-readable catalogue of public cancer imaging datasets.

Built from the project's dataset survey. The failure loop (``loop/failures.py``)
uses it to recommend *which* data to acquire next when a slice of cases fails.

Every field is as surveyed and **must be verified against the custodian's page
(TCIA / IDC / challenge site) before use**. Licences and subject counts change
between releases. ``notes`` records known discrepancies found while encoding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    source: str
    domain: str  # radiology | pathology | dermoscopy | endoscopy | ultrasound
    modalities: tuple[str, ...]
    organs: tuple[str, ...]
    subjects: str
    longitudinal: bool
    longitudinal_note: str = ""
    reports: str = ""
    access: str = ""
    license: str = ""
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self):
        return asdict(self)


_R, _P = "radiology", "pathology"

CATALOGUE: tuple[DatasetEntry, ...] = (
    DatasetEntry("Head-Neck Cetuximab (RTOG 0522)", "TCIA / NRG-RTOG", _R, ("PET", "CT", "RTSTRUCT"), ("head_neck",), "111",
                 True, "pre- and post-treatment FDG PET/CT with RT plans", "RT plans, structure sets; trial clinical data",
                 "TCIA registration", "TCIA collection licence (check page)", tags=("treatment_response",)),
    DatasetEntry("NSCLC-Radiomics", "TCIA (MAASTRO)", _R, ("CT", "RTSTRUCT", "SEG"), ("lung",), "422", False,
                 "baseline only", "clinical attributes incl. survival; segmentations", "TCIA", "CC BY",
                 notes="Custodian is MAASTRO Clinic (Maastricht); survey listed MD Anderson - verify."),
    DatasetEntry("LIDC-IDRI", "TCIA / LIDC consortium", _R, ("CT",), ("lung",), "1010", False, "single timepoint",
                 "nodule contours by up to 4 radiologists (XML), malignancy ratings", "TCIA", "CC BY",
                 tags=("nodule", "localisation")),
    DatasetEntry("LungCT-Diagnosis", "TCIA", _R, ("CT",), ("lung",), "61", False, "pre-operative",
                 "pathology stage, survival", "TCIA", "CC BY"),
    DatasetEntry("NLST", "NCI / IDC", _R, ("CT",), ("lung",), "~26,000", True,
                 "baseline + up to 2 annual screening CTs with follow-up outcomes",
                 "demographics, smoking, cancer outcome and pathology", "IDC / CDAS request", "see NLST terms",
                 notes="Future-risk precedent: Sybil (Mikhael et al., JCO 2023) was trained on NLST.",
                 tags=("future_risk", "screening")),
    DatasetEntry("Anti-PD-1 Lung", "TCIA", _R, ("CT", "PET", "SEG"), ("lung",), "46", True,
                 "pre/post immunotherapy", "", "TCIA", "CC BY", tags=("treatment_response",)),
    DatasetEntry("Lung-PET-CT-Dx", "TCIA", _R, ("CT", "PET"), ("lung",), "355", False, "single PET/CT",
                 "tumour bounding boxes, histology type", "TCIA", "CC BY",
                 notes="Survey listed it twice and as 'Sybil'; Sybil was trained on NLST, not this set."),
    DatasetEntry("RIDER Lung CT", "TCIA", _R, ("CT",), ("lung",), "32", True, "same-day coffee-break repeat scans",
                 "lesion measurements", "TCIA", "CC BY", tags=("test_retest",)),
    DatasetEntry("RIDER Lung PET-CT", "TCIA", _R, ("PET", "CT"), ("lung",), "244", True, "repeat PET/CT", "",
                 "TCIA", "CC BY", tags=("test_retest",)),
    DatasetEntry("RIDER Neuro MRI", "TCIA", _R, ("MRI",), ("brain",), "19", True, "repeat MRI 1-2 days apart",
                 "DTI, DCE", "TCIA", "CC BY", tags=("test_retest",)),
    DatasetEntry("RIDER Breast MRI", "TCIA", _R, ("MRI",), ("breast",), "5", True, "two scans ~15 min apart", "",
                 "TCIA", "CC BY", tags=("test_retest",)),
    DatasetEntry("I-SPY 1 (ACRIN 6657)", "TCIA / ACRIN", _R, ("MRI",), ("breast",), "222", True,
                 "serial MRI during neoadjuvant chemotherapy", "clinical and outcome tables", "TCIA", "CC BY",
                 notes="Survey said baseline only; I-SPY 1 has up to 4 MRI timepoints - verify.",
                 tags=("treatment_response",)),
    DatasetEntry("NaF Prostate", "TCIA", _R, ("PET", "CT"), ("prostate", "bone"), "9", True,
                 "baseline and follow-up", "", "TCIA", "CC BY"),
    DatasetEntry("QIN-Breast", "TCIA / QIN", _R, ("PET", "CT", "MRI"), ("breast",), "68", True,
                 "longitudinal during neoadjuvant therapy", "treatment/response", "TCIA", "CC BY",
                 tags=("treatment_response",)),
    DatasetEntry("ACRIN-FLT-Breast", "TCIA / ACRIN", _R, ("PET", "CT"), ("breast",), "83", True,
                 "pre/post therapy", "treatment data", "TCIA", "CC BY", tags=("treatment_response",)),
    DatasetEntry("Breast-Cancer-Screening-DBT", "TCIA (Duke)", _R, ("DBT",), ("breast",), "5,060", False,
                 "screening exams", "lesion boxes for a subset", "TCIA", "CC BY", tags=("screening",)),
    DatasetEntry("C4KC-KiTS", "TCIA / KiTS19", _R, ("CT", "SEG"), ("kidney",), "210", False, "", "segmentations",
                 "TCIA", "CC BY"),
    DatasetEntry("TCGA-BRCA", "NCI TCGA / TCIA / IDC", _R, ("MRI", "MG", "SM"), ("breast",), "1,098 (cases)", False,
                 "", "clinical, genomics, whole-slide images", "open imaging; controlled genomics (dbGaP)", "mixed",
                 tags=("radiology_pathology_link", "genomics")),
    DatasetEntry("TCGA-OV", "NCI TCGA", _R, ("CT", "MRI", "SM"), ("ovary",), "591", False, "",
                 "clinical, genomics, WSIs", "open imaging; controlled genomics", "mixed", tags=("genomics",)),
    DatasetEntry("TCGA-KIRC", "NCI TCGA", _R, ("CT", "MRI", "SM"), ("kidney",), "537", False, "",
                 "clinical, genomics, WSIs", "open imaging; controlled genomics", "mixed", tags=("genomics",)),
    DatasetEntry("TCGA-GBM", "NCI TCGA", _R, ("MRI", "CT", "SM"), ("brain",), "607", False, "",
                 "clinical, genomics, WSIs", "open imaging; controlled genomics", "mixed", tags=("genomics",)),
    DatasetEntry("TCGA-LGG", "NCI TCGA", _R, ("MRI", "CT", "SM"), ("brain",), "516", False, "",
                 "clinical, genomics, WSIs", "open imaging; controlled genomics", "mixed", tags=("genomics",)),
    DatasetEntry("TCGA-LIHC", "NCI TCGA", _R, ("CT", "MRI", "SM"), ("liver",), "377", False, "",
                 "clinical, genomics, WSIs", "open imaging; controlled genomics", "mixed", tags=("genomics",)),
    DatasetEntry("Lung-Fused-CT-Pathology", "TCIA", _R, ("CT", "SM"), ("lung",), "6", False,
                 "CT co-registered with pathology", "fusion data", "TCIA", "CC BY",
                 tags=("radiology_pathology_link",)),
    DatasetEntry("CAMELYON16/17", "Radboud UMC & UMC Utrecht (grand-challenge.org)", _P, ("SM",), ("breast", "lymph_node"),
                 "400 WSIs / 1,000 WSIs", False, "", "pathologist metastasis contours", "public challenge", "CC0 / see site",
                 notes="Survey listed 'NKI/UMC (Kaggle)' - verify custodian.", tags=("localisation", "metastasis")),
    DatasetEntry("PanNuke", "Gamper et al. 2019/2020", _P, ("H&E",), ("multi",), "~7,900 patches, 19 tissues", False,
                 "", "nucleus instance masks and 5 nucleus types", "public", "CC BY-NC-SA",
                 notes="Survey said 25 types/CC BY - verify.", tags=("nuclei", "cell_level")),
    DatasetEntry("BreakHis", "Spanhol et al. 2016 (UFPR, Brazil)", _P, ("H&E",), ("breast",), "7,909 images / 82 patients",
                 False, "", "benign/malignant + 8 subtypes", "request form", "research use",
                 notes="Survey listed PUCRS; the original paper is from UFPR - verify."),
    DatasetEntry("Kather CRC (NCT-CRC-HE-100K)", "Kather et al. 2019 (Zenodo)", _P, ("H&E",), ("colon",),
                 "100,000 patches", False, "", "9 tissue classes", "public", "CC BY", tags=("tissue_class",)),
    DatasetEntry("BACH", "ICIAR 2018 challenge", _P, ("H&E",), ("breast",), "400 images (+WSIs)", False, "",
                 "normal/benign/in situ/invasive", "public challenge", "CC BY-NC-ND (check)"),
    DatasetEntry("PatchCamelyon (PCam)", "Veeling et al. MICCAI 2018", _P, ("H&E",), ("breast", "lymph_node"),
                 "327,680 patches", False, "", "binary metastasis label", "public", "CC0",
                 notes="The PCam paper introduced rotation-equivariant CNNs for pathology (cf. models/equivariant.py).",
                 tags=("metastasis",)),
    DatasetEntry("HAM10000 / ISIC Archive", "ViDIR Vienna / ISIC", "dermoscopy", ("dermoscopy",), ("skin",),
                 "~10,000 images", False, "", "7 diagnostic classes, many histology-confirmed", "public",
                 "CC BY-NC", tags=("melanoma",)),
    DatasetEntry("CBIS-DDSM", "TCIA", _R, ("MG",), ("breast",), "~1,566 patients", False, "",
                 "ROI masks, BI-RADS, pathology-verified labels", "TCIA", "CC BY", tags=("screening",)),
    DatasetEntry("PROSTATEx", "TCIA / SPIE-AAPM-NCI", _R, ("MRI",), ("prostate",), "346", False, "",
                 "lesion locations, clinical significance", "TCIA", "CC BY"),
    DatasetEntry("ACRIN 6664 CT Colonography", "TCIA / ACRIN", _R, ("CT",), ("colon",), "825", False,
                 "prone/supine pair", "polyp findings", "TCIA", "CC BY", tags=("screening",)),
    DatasetEntry("BraTS", "RSNA-ASNR-MICCAI", _R, ("MRI",), ("brain",), "~2,000 (2021+)", False,
                 "pre-operative multi-parametric MRI", "tumour sub-region segmentations", "challenge registration",
                 "see challenge terms", tags=("localisation",)),
    DatasetEntry("LUMIERE", "Suter et al. 2022 (Sci. Data)", _R, ("MRI",), ("brain",), "91", True,
                 "post-treatment glioma follow-up MRIs", "RANO response labels", "public registration",
                 "see page", notes="Longitudinal glioblastoma follow-up; verify availability.", tags=("treatment_response",)),
    DatasetEntry("Kvasir-SEG / CVC-ClinicDB", "Simula / CVC", "endoscopy", ("endoscopy",), ("colon",),
                 "1,000 + 612 images", False, "", "polyp masks", "public", "research use"),
    DatasetEntry("BUSI", "Al-Dhabyani et al. 2020", "ultrasound", ("US",), ("breast",), "780 images", False, "",
                 "benign/malignant/normal + masks", "public", "CC0 (check)"),
)


def search(organ: str | None = None, modality: str | None = None, domain: str | None = None,
           longitudinal: bool | None = None, tag: str | None = None) -> list[DatasetEntry]:
    out = []
    for d in CATALOGUE:
        if organ and organ not in d.organs and "multi" not in d.organs:
            continue
        if modality and not any(modality.lower() == m.lower() for m in d.modalities):
            continue
        if domain and d.domain != domain:
            continue
        if longitudinal is not None and d.longitudinal != longitudinal:
            continue
        if tag and tag not in d.tags:
            continue
        out.append(d)
    return out


MODALITY_ALIASES = {"histology": ("H&E", "SM"), "mri": ("MRI",), "ct": ("CT",), "pet": ("PET",),
                    "mammography": ("MG", "DBT"), "ultrasound": ("US",), "dermoscopy": ("dermoscopy",)}
