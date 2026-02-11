# MENINGIOMA CLINICAL DECISION GRAPH: COMPLETE STRUCTURE SPECIFICATION

**Version**: 2.0 (Evidence-Based Clinical Thresholds)  
**Date**: February 10, 2026  
**Purpose**: Complete technical specification for extracting Markov Decision Process (MDP) graphs from longitudinal meningioma clinical notes

---

## TABLE OF CONTENTS

1. [Overview & Architecture](#1-overview--architecture)
2. [Stratification Layer (Static Factors)](#2-stratification-layer-static-factors)
3. [State Space (Dynamic Variables)](#3-state-space-dynamic-variables)
4. [Action Space](#4-action-space)
5. [Transition Structure](#5-transition-structure)
6. [Computational Definitions](#6-computational-definitions)
7. [Data Flow Pipeline](#7-data-flow-pipeline)
8. [Edge Cases & Validation Rules](#8-edge-cases--validation-rules)
9. [Output Format Specification](#9-output-format-specification)

---

## 1. OVERVIEW & ARCHITECTURE

### 1.1 High-Level Structure

This is a **multi-graph MDP system** where:
- **90 separate directed graphs** exist (one per stratification key)
- Each graph represents a **finite-state Markov Decision Process**
- **Standard MDP formulation**: `(S, A, T, R)` where:
  - `S` = State space (108 states per graph)
  - `A` = Action space (8 actions)
  - `T` = Transition function: `(state, action) → next_state`
  - `R` = Reward/outcome (optional, for future Q-value computation)

### 1.2 Design Principles

1. **Markov Property**: Current state fully characterizes patient condition
2. **No Action × Attribute Crossing**: Treatment phase is IN state, not on edge
3. **Bucketed Actions**: Observation intervals are bucketed (short/medium/long), not fixed
4. **Clinical Thresholds**: All bucket boundaries match evidence-based decision points
5. **Natural Sparsity**: Treatment phases constrain available actions
6. **Cross-Graph Transitions**: Grade changes trigger transitions between graphs

### 1.3 Complexity Summary

```
Stratification trees: 90
States per tree: 108
Total state space: 9,720 states
Actions: 8
Avg actions per state: ~4 (with phase constraints)
Expected edges: ~17,280 (before cross-graph transitions)

Data:
- 50 patients × 35 avg visits = 1,750 observations
- Coverage: ~18% of state space
- Dense subgraphs expected in: Grade 1, convexity, small stable tumors
```

---

## 2. STRATIFICATION LAYER (STATIC FACTORS)

### 2.1 Overview

The stratification layer creates **90 separate trees**, one for each unique combination of patient demographics and tumor characteristics that remain **constant throughout a patient's care**.

### 2.2 Stratification Variables

#### 2.2.1 Age at Diagnosis (3 buckets)

```python
AGE_BUCKETS = {
    '<50': {
        'range': [0, 50),
        'clinical_significance': 'Younger patients, higher growth rates, longer surveillance needed'
    },
    '50-65': {
        'range': [50, 65),
        'clinical_significance': 'Middle-aged cohort, standard risk profile'
    },
    '≥65': {
        'range': [65, ∞),
        'clinical_significance': 'Elderly, higher surgical risk, more conservative management'
    }
}
```

**Computation**:
```python
def compute_age_bucket(age_at_diagnosis: float) -> str:
    if age_at_diagnosis < 50:
        return '<50'
    elif age_at_diagnosis < 65:
        return '50-65'
    else:
        return '≥65'
```

**Source**: `age_at_diagnosis` extracted from initial clinical note or demographics

---

#### 2.2.2 Gender (2 buckets)

```python
GENDER_BUCKETS = {
    'M': {
        'values': ['male', 'M', 'man'],
        'clinical_significance': 'Lower incidence, potentially higher grade risk'
    },
    'F': {
        'values': ['female', 'F', 'woman'],
        'clinical_significance': 'Higher incidence, hormone-related growth patterns'
    }
}
```

**Computation**:
```python
def compute_gender_bucket(gender: str) -> str:
    gender_lower = gender.lower().strip()
    if gender_lower in ['male', 'm', 'man']:
        return 'M'
    elif gender_lower in ['female', 'f', 'woman']:
        return 'F'
    else:
        raise ValueError(f"Unknown gender: {gender}")
```

**Source**: `gender` extracted from demographics

---

#### 2.2.3 Tumor Grade (3 buckets)

```python
GRADE_BUCKETS = {
    'grade_1': {
        'who_grade': 1,
        'synonyms': ['benign', 'WHO grade I', 'grade 1', 'G1'],
        'clinical_significance': 'Most common (80%), slow growth, low recurrence (~7-15%)',
        'typical_growth_rate': '1-2 mm/year'
    },
    'grade_2': {
        'who_grade': 2,
        'synonyms': ['atypical', 'WHO grade II', 'grade 2', 'G2'],
        'clinical_significance': 'Atypical (15-20%), faster growth, higher recurrence (~30-40%)',
        'typical_growth_rate': '3-5 mm/year'
    },
    'grade_3': {
        'who_grade': 3,
        'synonyms': ['malignant', 'anaplastic', 'WHO grade III', 'grade 3', 'G3'],
        'clinical_significance': 'Rare (1-4%), aggressive, very high recurrence (>70%)',
        'typical_growth_rate': '>5 mm/year'
    }
}
```

**Computation**:
```python
def compute_grade_bucket(pathology_text: str) -> str:
    """
    Extract WHO grade from pathology report.
    CRITICAL: Use canonicalized grade to prevent false transitions.
    """
    text_lower = pathology_text.lower()
    
    # Grade 3 patterns
    if any(word in text_lower for word in ['grade 3', 'grade iii', 'g3', 'malignant', 'anaplastic']):
        return 'grade_3'
    
    # Grade 2 patterns
    if any(word in text_lower for word in ['grade 2', 'grade ii', 'g2', 'atypical']):
        return 'grade_2'
    
    # Grade 1 patterns (default for meningioma)
    if any(word in text_lower for word in ['grade 1', 'grade i', 'g1', 'benign']):
        return 'grade_1'
    
    # Default: assume grade 1 if no pathology available (most common)
    return 'grade_1'
```

**Source**: Pathology report (usually available only after surgery or biopsy)

**⚠️ CRITICAL GRADE HANDLING**: 
- Grade is initially **unknown** until pathology obtained
- Before surgery: Use `grade_unknown` or impute from imaging (see Section 8.3)
- After surgery: Use pathology-confirmed grade
- **Cross-graph transition**: If grade changes (e.g., upgrade on recurrence), patient moves to different stratification graph

---

#### 2.2.4 Tumor Location (5 buckets)

```python
LOCATION_BUCKETS = {
    'convexity': {
        'description': 'Surface of brain directly under skull',
        'frequency': '~20-25% of meningiomas',
        'surgical_accessibility': 'High - relatively straightforward surgery',
        'recurrence_rate': '5-15% after gross total resection',
        'clinical_significance': 'May not cause symptoms until large; prone to seizures'
    },
    'skull_base': {
        'description': 'Base of skull, near cranial nerves',
        'frequency': '~20-30% of meningiomas',
        'surgical_accessibility': 'Low - difficult, high morbidity risk',
        'recurrence_rate': '26.5% (higher than convexity)',
        'clinical_significance': 'Early symptoms (vision, hearing), difficult resection'
    },
    'parasagittal': {
        'description': 'Adjacent to falx cerebri (midline)',
        'frequency': '~15-20% of meningiomas',
        'surgical_accessibility': 'Medium - involves superior sagittal sinus',
        'recurrence_rate': '10-15%',
        'clinical_significance': 'Leg weakness, personality changes, venous involvement'
    },
    'sphenoid_wing': {
        'description': 'Lateral skull base behind eyes',
        'frequency': '~10-15% of meningiomas',
        'surgical_accessibility': 'Medium - involves optic nerve, carotid',
        'recurrence_rate': 'Variable',
        'clinical_significance': 'Vision problems, proptosis'
    },
    'other': {
        'description': 'Intraventricular, olfactory groove, tentorial, spinal, etc.',
        'frequency': '~25-30% of meningiomas',
        'surgical_accessibility': 'Highly variable',
        'recurrence_rate': 'Variable',
        'clinical_significance': 'Heterogeneous group'
    }
}
```

**Computation**:
```python
def compute_location_bucket(location_text: str) -> str:
    """
    Extract anatomical location from radiology or clinical notes.
    """
    text_lower = location_text.lower()
    
    # Convexity
    if 'convexity' in text_lower or 'convex' in text_lower:
        return 'convexity'
    
    # Skull base
    if any(term in text_lower for term in ['skull base', 'cavernous sinus', 
                                            'petroclival', 'clivus', 'foramen magnum']):
        return 'skull_base'
    
    # Parasagittal/Falcine
    if any(term in text_lower for term in ['parasagittal', 'falx', 'falcine', 
                                            'sagittal sinus']):
        return 'parasagittal'
    
    # Sphenoid wing
    if 'sphenoid' in text_lower and 'wing' in text_lower:
        return 'sphenoid_wing'
    
    # Default: other
    return 'other'
```

**Source**: Radiology report (MRI) or operative note

---

### 2.3 Stratification Key Construction

Each patient is assigned a **stratification key** at diagnosis that determines which of the 90 graphs they belong to.

```python
class StratificationKey:
    """
    Immutable identifier for a patient's stratification graph.
    """
    age_bucket: str      # '<50', '50-65', '≥65'
    gender: str          # 'M', 'F'
    grade: str           # 'grade_1', 'grade_2', 'grade_3'
    location: str        # 'convexity', 'skull_base', etc.
    
    def __str__(self):
        return f"{self.age_bucket}_{self.gender}_{self.grade}_{self.location}"
    
    def __hash__(self):
        return hash((self.age_bucket, self.gender, self.grade, self.location))

# Example:
# StratificationKey('<50', 'F', 'grade_1', 'convexity')
# → String representation: "<50_F_grade_1_convexity"
```

**Total stratification keys**: 3 (age) × 2 (gender) × 3 (grade) × 5 (location) = **90 graphs**

---

## 3. STATE SPACE (DYNAMIC VARIABLES)

### 3.1 Overview

Each stratification graph contains **108 states** defined by 4 dynamic clinical variables that change over time during a patient's longitudinal care.

### 3.2 State Variables

#### 3.2.1 Tumor Size (3 buckets)

```python
TUMOR_SIZE_BUCKETS = {
    'small': {
        'range_cm': [0, 3.0),
        'diameter_mm': [0, 30),
        'clinical_significance': 'Observation appropriate; SRS eligible; low symptom risk',
        'typical_management': 'Watch and wait with annual MRI'
    },
    'medium': {
        'range_cm': [3.0, 5.0),
        'diameter_mm': [30, 50),
        'clinical_significance': 'Increased symptom risk; surgery often considered',
        'typical_management': 'Increased surveillance or intervention'
    },
    'large': {
        'range_cm': [5.0, ∞),
        'diameter_mm': [50, ∞),
        'clinical_significance': 'Giant tumors; complex surgery; high morbidity risk',
        'typical_management': 'Surgical resection usually indicated'
    }
}
```

**Evidence basis**:
- **3cm threshold**: Risk of symptom progression increases significantly; SRS eligibility cutoff
- **5cm threshold**: Classified as "giant meningioma"; substantially increased surgical complexity

**Computation**:
```python
def compute_tumor_size_bucket(diameter_cm: float) -> str:
    """
    Convert tumor diameter (maximum dimension in cm) to bucket.
    
    Args:
        diameter_cm: Maximum tumor diameter in centimeters
    
    Returns:
        'small', 'medium', or 'large'
    """
    if diameter_cm < 3.0:
        return 'small'
    elif diameter_cm < 5.0:
        return 'medium'
    else:
        return 'large'
```

**Source**: MRI report, usually reported as maximum diameter

**Measurement notes**:
- Use **maximum diameter** in any plane
- If volume reported: convert using `diameter = (6 × volume / π)^(1/3)`
- If dimensions given (e.g., "2.3 × 1.8 × 2.1 cm"): use maximum (2.3 cm)

---

#### 3.2.2 Symptoms (2 buckets)

```python
SYMPTOM_BUCKETS = {
    'none': {
        'description': 'Asymptomatic / incidental finding',
        'clinical_significance': 'Observation often appropriate',
        'examples': 'Tumor discovered on imaging for unrelated reason'
    },
    'present': {
        'description': 'Any neurological symptoms attributable to tumor',
        'clinical_significance': 'Symptomatic tumors typically require intervention',
        'examples': [
            'Headaches',
            'Seizures',
            'Motor weakness',
            'Sensory deficits',
            'Visual changes',
            'Cognitive changes',
            'Speech difficulties',
            'Cranial nerve palsies'
        ]
    }
}
```

**Computation**:
```python
def compute_symptom_bucket(clinical_note: str) -> str:
    """
    Determine if patient is symptomatic from tumor.
    
    IMPORTANT: Only count symptoms ATTRIBUTABLE to tumor, not incidental findings.
    """
    note_lower = clinical_note.lower()
    
    # Asymptomatic keywords
    asymptomatic_patterns = [
        'asymptomatic',
        'incidental finding',
        'no symptoms',
        'discovered incidentally',
        'found on imaging for',
        'no complaints'
    ]
    
    # Symptomatic keywords
    symptomatic_patterns = [
        'headache', 'seizure', 'weakness', 'numbness',
        'vision change', 'visual deficit', 'diplopia',
        'speech difficulty', 'aphasia', 'cognitive',
        'personality change', 'gait difficulty',
        'cranial nerve', 'facial weakness'
    ]
    
    # Check for asymptomatic patterns
    if any(pattern in note_lower for pattern in asymptomatic_patterns):
        return 'none'
    
    # Check for symptomatic patterns
    if any(pattern in note_lower for pattern in symptomatic_patterns):
        return 'present'
    
    # Default: assume asymptomatic if not specified
    return 'none'
```

**Source**: Clinical notes (history of present illness, review of systems)

---

#### 3.2.3 Growth Velocity (3 buckets)

```python
GROWTH_VELOCITY_BUCKETS = {
    'stable': {
        'range_mm_per_year': [float('-inf'), 2.0),
        'range_cm_per_year': [float('-inf'), 0.2),
        'clinical_significance': 'Typical slow growth or stable; observation appropriate',
        'who_grade_association': 'Usually Grade 1',
        'management': 'Annual surveillance MRI sufficient'
    },
    'slow_growth': {
        'range_mm_per_year': [2.0, 5.0),
        'range_cm_per_year': [0.2, 0.5),
        'clinical_significance': 'Standard growing meningioma; increased monitoring',
        'who_grade_association': 'Usually Grade 1, sometimes Grade 2',
        'management': '6-month surveillance; consider intervention if symptomatic'
    },
    'fast_growth': {
        'range_mm_per_year': [5.0, ∞),
        'range_cm_per_year': [0.5, ∞),
        'clinical_significance': 'Concerning rapid growth; often higher grade',
        'who_grade_association': 'Often Grade 2-3',
        'management': 'Urgent intervention typically needed'
    }
}
```

**Evidence basis**:
- **2mm/year threshold**: Typical slow meningioma growth rate (1-2mm/year is standard)
- **5mm/year threshold**: Distinguishes standard growth from aggressive/atypical behavior

**Computation**:
```python
def compute_growth_velocity_bucket(
    current_size_cm: float,
    previous_size_cm: float,
    time_interval_months: float
) -> str:
    """
    Calculate annualized growth velocity from two measurements.
    
    Args:
        current_size_cm: Current tumor diameter (cm)
        previous_size_cm: Previous tumor diameter (cm)
        time_interval_months: Time between measurements (months)
    
    Returns:
        'stable', 'slow_growth', or 'fast_growth'
    
    Notes:
        - Requires at least 2 measurements
        - Negative growth (shrinkage) → 'stable'
        - Time interval should be ≥3 months for reliability
    """
    if time_interval_months <= 0:
        raise ValueError("Time interval must be positive")
    
    # Calculate annualized growth rate
    size_change_cm = current_size_cm - previous_size_cm
    annualized_growth_cm = (size_change_cm / time_interval_months) * 12.0
    annualized_growth_mm = annualized_growth_cm * 10.0
    
    # Handle negative growth (shrinkage) → stable
    if annualized_growth_mm < 0:
        annualized_growth_mm = 0
    
    # Bucket assignment
    if annualized_growth_mm < 2.0:
        return 'stable'
    elif annualized_growth_mm < 5.0:
        return 'slow_growth'
    else:
        return 'fast_growth'
```

**Source**: Sequential MRI reports with tumor measurements

**Special cases**:
- **First visit**: No prior measurement → cannot compute → assign `'stable'` as default
- **Missing measurements**: Use last known size, impute forward (see Section 8.1)
- **Post-surgery**: Reset growth computation using new baseline size

---

#### 3.2.4 Treatment Phase (6 buckets)

```python
TREATMENT_PHASE_BUCKETS = {
    'naive': {
        'description': 'No prior treatment',
        'clinical_significance': 'All treatment options available',
        'typical_duration': 'Until first intervention',
        'available_actions': ['observe_short', 'observe_medium', 'observe_long',
                             'surgery_gtr', 'surgery_str', 'radiation_srs', 
                             'radiation_fsrt', 'supportive_care']
    },
    'early_postop': {
        'description': '0-6 months after surgery',
        'clinical_significance': 'High complication risk; aggressive surveillance needed',
        'typical_duration': '6 months',
        'available_actions': ['observe_short', 'observe_medium', 'supportive_care'],
        'surveillance_interval': '3 months typical'
    },
    'late_postop': {
        'description': '>6 months after surgery',
        'clinical_significance': 'Routine post-operative surveillance',
        'typical_duration': 'Until recurrence or lifelong',
        'available_actions': ['observe_medium', 'observe_long', 'supportive_care'],
        'surveillance_interval': '6-12 months typical'
    },
    'early_postrad': {
        'description': '0-6 months after radiation therapy',
        'clinical_significance': 'Monitoring for radiation response and toxicity',
        'typical_duration': '6 months',
        'available_actions': ['observe_short', 'observe_medium', 'supportive_care'],
        'surveillance_interval': '3 months typical'
    },
    'late_postrad': {
        'description': '>6 months after radiation therapy',
        'clinical_significance': 'Long-term surveillance for tumor control',
        'typical_duration': 'Until recurrence or lifelong',
        'available_actions': ['observe_medium', 'observe_long', 'supportive_care'],
        'surveillance_interval': '6-12 months typical'
    },
    'recurrent': {
        'description': 'Tumor recurrence detected after prior treatment',
        'clinical_significance': 'Salvage therapy options; higher risk',
        'typical_duration': 'Until retreatment',
        'available_actions': ['surgery_str', 'radiation_srs', 'radiation_fsrt', 
                             'supportive_care'],
        'note': 'Gross total resection (GTR) often not feasible at recurrence'
    }
}
```

**Computation**:
```python
def compute_treatment_phase(
    visit_date_months: float,
    surgery_date_months: Optional[float],
    radiation_date_months: Optional[float],
    recurrence_detected: bool
) -> str:
    """
    Determine treatment phase based on intervention history.
    
    Args:
        visit_date_months: Months since diagnosis for current visit
        surgery_date_months: Months since diagnosis when surgery occurred (None if no surgery)
        radiation_date_months: Months since diagnosis when radiation occurred (None if no radiation)
        recurrence_detected: Boolean indicating if recurrence has been detected
    
    Returns:
        Treatment phase string
    
    Priority order:
        1. Recurrent (if recurrence detected)
        2. Early post-op (if surgery within 6 months)
        3. Late post-op (if surgery >6 months ago)
        4. Early post-rad (if radiation within 6 months)
        5. Late post-rad (if radiation >6 months ago)
        6. Naive (if no treatment)
    """
    # Priority 1: Recurrence overrides everything
    if recurrence_detected:
        return 'recurrent'
    
    # Priority 2: Surgery phases
    if surgery_date_months is not None:
        months_since_surgery = visit_date_months - surgery_date_months
        if months_since_surgery <= 6:
            return 'early_postop'
        else:
            return 'late_postop'
    
    # Priority 3: Radiation phases
    if radiation_date_months is not None:
        months_since_radiation = visit_date_months - radiation_date_months
        if months_since_radiation <= 6:
            return 'early_postrad'
        else:
            return 'late_postrad'
    
    # Default: Treatment naive
    return 'naive'
```

**Source**: 
- Surgery date: Operative note
- Radiation date: Radiation oncology note
- Recurrence: Radiology report + clinical note

**⚠️ CRITICAL PHASE TRANSITIONS**:
- Phases are **deterministic** given intervention dates
- Phase transitions are **automatic** at 6-month mark
- **Cross-treatment sequences**: If patient has both surgery AND radiation, surgery takes priority (most recent)

---

### 3.3 State ID Construction

Each state is uniquely identified by concatenating all 4 dynamic variables:

```python
class StateID:
    """
    Unique identifier for a state within a stratification graph.
    """
    tumor_size: str           # 'small', 'medium', 'large'
    symptoms: str             # 'none', 'present'
    growth_velocity: str      # 'stable', 'slow_growth', 'fast_growth'
    treatment_phase: str      # 6 possible values
    
    def __str__(self):
        return f"{self.tumor_size}_{self.symptoms}_{self.growth_velocity}_{self.treatment_phase}"
    
    def __hash__(self):
        return hash((self.tumor_size, self.symptoms, self.growth_velocity, self.treatment_phase))

# Example:
# StateID('small', 'none', 'stable', 'naive')
# → String representation: "small_none_stable_naive"
```

**Total states per graph**: 3 (size) × 2 (symptoms) × 3 (growth) × 6 (phase) = **108 states**

---

## 4. ACTION SPACE

### 4.1 Overview

8 actions are available, representing the clinical decisions a physician can make at each visit. **Not all actions are available in all states** (see Section 4.3 for constraints).

### 4.2 Action Definitions

#### 4.2.1 Observation Actions (3 buckets)

```python
OBSERVATION_ACTIONS = {
    'observe_short': {
        'interval_range_months': [0, 4.5),
        'target_interval_months': 3,
        'clinical_meaning': 'Aggressive surveillance',
        'typical_contexts': [
            'Post-operative (<6 months)',
            'Growing tumor',
            'New symptoms',
            'Grade 2-3 tumors'
        ]
    },
    'observe_medium': {
        'interval_range_months': [4.5, 9.0),
        'target_interval_months': 6,
        'clinical_meaning': 'Standard surveillance',
        'typical_contexts': [
            'Stable small tumors',
            'Post-operative (>6 months)',
            'Grade 1 tumors'
        ]
    },
    'observe_long': {
        'interval_range_months': [9.0, ∞),
        'target_interval_months': 12,
        'clinical_meaning': 'Routine surveillance',
        'typical_contexts': [
            'Very stable tumors',
            'Late post-operative (>2 years)',
            'Elderly patients',
            'Small calcified tumors'
        ]
    }
}
```

**Observation interval inference**:
```python
def infer_observation_action(
    current_visit_months: float,
    next_visit_months: float
) -> str:
    """
    Infer observation action from actual visit interval.
    
    Args:
        current_visit_months: Months since diagnosis for current visit
        next_visit_months: Months since diagnosis for next visit
    
    Returns:
        'observe_short', 'observe_medium', or 'observe_long'
    """
    interval_months = next_visit_months - current_visit_months
    
    if interval_months < 4.5:
        return 'observe_short'
    elif interval_months < 9.0:
        return 'observe_medium'
    else:
        return 'observe_long'
```

**Rationale for bucketing**:
- Clinical intent: "See in 6 months" (not "see in exactly 6.0 months")
- Actual variability: 6-month follow-ups occur at 5.2-7.1 months in practice
- Statistical power: Pools similar decisions for analysis
- Counterfactual meaning: "Short vs long surveillance" is clinically meaningful; "6.0mo vs 6.3mo" is not

---

#### 4.2.2 Surgical Actions (2 types)

```python
SURGICAL_ACTIONS = {
    'surgery_gtr': {
        'full_name': 'Gross Total Resection',
        'description': 'Complete surgical removal of tumor',
        'simpson_grade': 'I or II',
        'clinical_significance': 'Lowest recurrence risk (5-15%)',
        'typical_indication': 'Accessible tumor, good surgical candidate',
        'expected_outcomes': {
            'tumor_size_change': 'large → small or small → minimal residual',
            'symptom_change': 'often present → none',
            'next_phase': 'early_postop'
        }
    },
    'surgery_str': {
        'full_name': 'Subtotal Resection',
        'description': 'Partial surgical removal (debulking)',
        'simpson_grade': 'III or IV',
        'clinical_significance': 'Higher recurrence risk (30-50%); used when GTR unsafe',
        'typical_indication': 'Skull base location, vessel involvement, high surgical risk',
        'expected_outcomes': {
            'tumor_size_change': 'large → medium or medium → small',
            'symptom_change': 'variable',
            'next_phase': 'early_postop'
        }
    }
}
```

**Extraction from operative note**:
```python
def extract_surgery_type(operative_note: str) -> str:
    """
    Determine if surgery was GTR or STR from operative note.
    """
    note_lower = operative_note.lower()
    
    # GTR patterns
    gtr_patterns = ['gross total resection', 'simpson grade i', 'simpson grade ii',
                    'complete resection', 'gtr', 'total removal']
    
    # STR patterns
    str_patterns = ['subtotal resection', 'simpson grade iii', 'simpson grade iv',
                    'partial resection', 'str', 'debulking', 'incomplete resection']
    
    if any(pattern in note_lower for pattern in gtr_patterns):
        return 'surgery_gtr'
    elif any(pattern in note_lower for pattern in str_patterns):
        return 'surgery_str'
    else:
        # Default: assume GTR if surgery occurred but not specified
        return 'surgery_gtr'
```

---

#### 4.2.3 Radiation Actions (2 types)

```python
RADIATION_ACTIONS = {
    'radiation_srs': {
        'full_name': 'Stereotactic Radiosurgery',
        'description': 'Single-session high-dose focal radiation (Gamma Knife, CyberKnife)',
        'typical_dose': '12-18 Gy single fraction',
        'clinical_significance': '90% tumor control for small tumors',
        'typical_indication': 'Small residual/recurrent tumors (<3cm), skull base',
        'expected_outcomes': {
            'tumor_size_change': 'stable or slight decrease',
            'symptom_change': 'variable',
            'next_phase': 'early_postrad'
        }
    },
    'radiation_fsrt': {
        'full_name': 'Fractionated Stereotactic Radiotherapy',
        'description': 'Multi-session focal radiation',
        'typical_dose': '45-54 Gy in 25-30 fractions',
        'clinical_significance': 'Used for larger tumors or near critical structures',
        'typical_indication': 'Tumors 3-5cm, near optic nerve/brainstem',
        'expected_outcomes': {
            'tumor_size_change': 'stable or slight decrease',
            'symptom_change': 'variable',
            'next_phase': 'early_postrad'
        }
    }
}
```

**Extraction from radiation oncology note**:
```python
def extract_radiation_type(radiation_note: str) -> str:
    """
    Determine if radiation was SRS or FSRT.
    """
    note_lower = radiation_note.lower()
    
    # SRS patterns
    srs_patterns = ['gamma knife', 'cyberknife', 'stereotactic radiosurgery',
                    'srs', 'single fraction', 'single session']
    
    # FSRT patterns
    fsrt_patterns = ['fractionated', 'fsrt', 'imrt', 'multiple fractions',
                     '25 fractions', '30 fractions']
    
    if any(pattern in note_lower for pattern in srs_patterns):
        return 'radiation_srs'
    elif any(pattern in note_lower for pattern in fsrt_patterns):
        return 'radiation_fsrt'
    else:
        # Default: assume SRS if radiation occurred but not specified
        return 'radiation_srs'
```

---

#### 4.2.4 Supportive Care

```python
SUPPORTIVE_CARE = {
    'supportive_care': {
        'description': 'Symptom management without disease-directed treatment',
        'clinical_significance': 'Used when intervention not feasible/appropriate',
        'typical_indication': [
            'Elderly with comorbidities',
            'Patient preference for no intervention',
            'End-of-life care',
            'Medically unfit for surgery/radiation'
        ],
        'interventions': [
            'Antiepileptic drugs for seizures',
            'Corticosteroids for edema',
            'Pain management',
            'Hospice referral'
        ],
        'expected_outcomes': {
            'tumor_size_change': 'none (no disease-directed therapy)',
            'symptom_change': 'symptomatic relief',
            'next_phase': 'no change'
        }
    }
}
```

---

### 4.3 Action Constraints by Treatment Phase

**Not all actions are available in all states.** Treatment phase determines which actions are clinically appropriate:

```python
PHASE_ACTION_CONSTRAINTS = {
    'naive': {
        'available_actions': [
            'observe_short', 'observe_medium', 'observe_long',
            'surgery_gtr', 'surgery_str',
            'radiation_srs', 'radiation_fsrt',
            'supportive_care'
        ],
        'rationale': 'All treatment options available for treatment-naive patient'
    },
    
    'early_postop': {
        'available_actions': [
            'observe_short', 'observe_medium',
            'supportive_care'
        ],
        'unavailable_actions': {
            'surgery_*': 'Too soon after surgery (healing period)',
            'radiation_*': 'Typically wait 6-12 weeks post-op',
            'observe_long': 'Aggressive surveillance needed early post-op'
        }
    },
    
    'late_postop': {
        'available_actions': [
            'observe_medium', 'observe_long',
            'supportive_care'
        ],
        'unavailable_actions': {
            'surgery_*': 'Included in recurrent phase if needed',
            'radiation_*': 'Included in recurrent phase if needed',
            'observe_short': 'Not needed for stable post-op patients'
        }
    },
    
    'early_postrad': {
        'available_actions': [
            'observe_short', 'observe_medium',
            'supportive_care'
        ],
        'unavailable_actions': {
            'surgery_*': 'Typically wait for radiation response',
            'radiation_*': 'Cannot re-radiate immediately',
            'observe_long': 'Close monitoring needed early post-radiation'
        }
    },
    
    'late_postrad': {
        'available_actions': [
            'observe_medium', 'observe_long',
            'supportive_care'
        ],
        'unavailable_actions': {
            'surgery_*': 'Included in recurrent phase if needed',
            'radiation_*': 'Cannot re-radiate to same area',
            'observe_short': 'Not needed for stable post-radiation patients'
        }
    },
    
    'recurrent': {
        'available_actions': [
            'surgery_str',  # Note: usually STR not GTR at recurrence
            'radiation_srs', 'radiation_fsrt',
            'supportive_care'
        ],
        'unavailable_actions': {
            'surgery_gtr': 'GTR rarely achievable at recurrence (scar tissue)',
            'observe_*': 'Active recurrence requires intervention'
        },
        'note': 'Salvage therapy options only'
    }
}
```

**Implementation**:
```python
def get_available_actions(treatment_phase: str) -> List[str]:
    """
    Return list of actions available in given treatment phase.
    """
    return PHASE_ACTION_CONSTRAINTS[treatment_phase]['available_actions']

def is_action_available(action: str, treatment_phase: str) -> bool:
    """
    Check if action is available in given treatment phase.
    """
    return action in PHASE_ACTION_CONSTRAINTS[treatment_phase]['available_actions']
```

**Rationale for constraints**:
- Reflects clinical reality (can't operate immediately post-op)
- Creates natural sparsity in graph
- Prevents nonsensical transitions
- Reduces state-action space from 108×8=864 to ~400-500 realistic transitions

---

## 5. TRANSITION STRUCTURE

### 5.1 Overview

Transitions represent the evolution of a patient's condition from one visit to the next, conditioned on the clinical decision (action) taken.

### 5.2 Transition Definition

```python
class Transition:
    """
    A single transition in the MDP graph.
    """
    # Source
    from_graph: StratificationKey
    from_state: StateID
    
    # Action
    action: str
    
    # Destination
    to_graph: StratificationKey
    to_state: StateID
    
    # Metadata
    count: int                    # Number of times observed
    patient_ids: Set[str]         # Which patients took this path
    time_elapsed_months: List[float]  # Actual time intervals
    outcomes: List[Dict]          # Complications, recurrence, etc.
    text_evidence: List[str]      # Supporting clinical notes
    
    # Special flags
    is_cross_graph: bool          # True if from_graph != to_graph
    changed_factors: List[str]    # Which stratification factors changed

# Standard transition (within same graph):
Transition(
    from_graph=StratificationKey('<50', 'F', 'grade_1', 'convexity'),
    from_state=StateID('small', 'none', 'stable', 'naive'),
    action='observe_medium',
    to_graph=StratificationKey('<50', 'F', 'grade_1', 'convexity'),
    to_state=StateID('small', 'none', 'stable', 'naive'),
    is_cross_graph=False
)

# Cross-graph transition (grade upgrade):
Transition(
    from_graph=StratificationKey('50-65', 'M', 'grade_1', 'skull_base'),
    from_state=StateID('medium', 'present', 'slow_growth', 'late_postop'),
    action='observe_medium',
    to_graph=StratificationKey('50-65', 'M', 'grade_2', 'skull_base'),
    to_state=StateID('medium', 'present', 'fast_growth', 'recurrent'),
    is_cross_graph=True,
    changed_factors=['tumor_grade']
)
```

### 5.3 Transition Construction Algorithm

```python
def build_transitions(patient_pathways: List[PatientPathway]) -> List[Transition]:
    """
    Build all transitions from patient visit sequences.
    
    Algorithm:
        For each patient:
            For each consecutive pair of visits (visit_i, visit_{i+1}):
                1. Compute from_state from visit_i variables
                2. Extract action from visit_i clinical note
                3. Compute to_state from visit_{i+1} variables
                4. Check if stratification changed (cross-graph)
                5. Create Transition object
                6. Aggregate by (from_graph, from_state, action, to_graph, to_state)
    
    Returns:
        List of unique transitions with counts and metadata
    """
    transitions = []
    
    for patient in patient_pathways:
        stratification_key = patient.stratification_key
        visits = patient.visits
        
        for i in range(len(visits) - 1):
            current_visit = visits[i]
            next_visit = visits[i + 1]
            
            # 1. Compute from_state
            from_state = StateID(
                tumor_size=current_visit.tumor_size_bucket,
                symptoms=current_visit.symptoms_bucket,
                growth_velocity=current_visit.growth_velocity_bucket,
                treatment_phase=current_visit.treatment_phase
            )
            
            # 2. Extract action
            action = extract_action(current_visit, next_visit)
            
            # 3. Compute to_state
            to_state = StateID(
                tumor_size=next_visit.tumor_size_bucket,
                symptoms=next_visit.symptoms_bucket,
                growth_velocity=next_visit.growth_velocity_bucket,
                treatment_phase=next_visit.treatment_phase
            )
            
            # 4. Check for stratification change
            from_graph = stratification_key
            to_graph = stratification_key
            is_cross_graph = False
            changed_factors = []
            
            if current_visit.grade_bucket != next_visit.grade_bucket:
                to_graph = StratificationKey(
                    age_bucket=stratification_key.age_bucket,
                    gender=stratification_key.gender,
                    grade=next_visit.grade_bucket,
                    location=stratification_key.location
                )
                is_cross_graph = True
                changed_factors.append('tumor_grade')
            
            # 5. Create transition
            transition = Transition(
                from_graph=from_graph,
                from_state=from_state,
                action=action,
                to_graph=to_graph,
                to_state=to_state,
                patient_ids={patient.patient_id},
                time_elapsed_months=[next_visit.months_since_diagnosis - current_visit.months_since_diagnosis],
                is_cross_graph=is_cross_graph,
                changed_factors=changed_factors
            )
            
            transitions.append(transition)
    
    # 6. Aggregate transitions
    return aggregate_transitions(transitions)
```

### 5.4 Action Extraction Logic

```python
def extract_action(current_visit: Visit, next_visit: Visit) -> str:
    """
    Infer action taken at current visit.
    
    Priority order:
        1. Explicit intervention (surgery, radiation)
        2. Supportive care only
        3. Observation (infer from interval)
    """
    # Check for surgery
    if current_visit.surgery_performed or next_visit.treatment_phase in ['early_postop']:
        if current_visit.simpson_grade in ['I', 'II']:
            return 'surgery_gtr'
        else:
            return 'surgery_str'
    
    # Check for radiation
    if current_visit.radiation_performed or next_visit.treatment_phase in ['early_postrad']:
        if current_visit.radiation_type == 'SRS':
            return 'radiation_srs'
        else:
            return 'radiation_fsrt'
    
    # Check for supportive care only
    if current_visit.supportive_care_only:
        return 'supportive_care'
    
    # Default: observation (infer interval)
    interval_months = next_visit.months_since_diagnosis - current_visit.months_since_diagnosis
    return infer_observation_action(current_visit.months_since_diagnosis, next_visit.months_since_diagnosis)
```

### 5.5 Cross-Graph Transitions

Cross-graph transitions occur when a patient's **stratification factors change** (almost always tumor grade).

**Triggers for cross-graph transition**:
1. **Grade upgrade on pathology**: Surgery reveals higher grade than suspected
2. **Malignant transformation**: Grade 1 → Grade 2 or Grade 3 on recurrence
3. **Grade downgrade**: Rare, but possible if initial biopsy was atypical area

**Handling**:
```python
# When grade changes, patient "moves" to new graph
if current_grade != next_grade:
    # Record as cross-graph transition
    transition.is_cross_graph = True
    transition.changed_factors = ['tumor_grade']
    
    # Update patient's stratification key for subsequent visits
    patient.stratification_key = StratificationKey(
        age_bucket=patient.stratification_key.age_bucket,
        gender=patient.stratification_key.gender,
        grade=next_grade,
        location=patient.stratification_key.location
    )
```

**Frequency**: Expected in <5% of transitions (grade changes are rare)

---

## 6. COMPUTATIONAL DEFINITIONS

### 6.1 Growth Velocity Computation

**Requirements**:
- Minimum 2 measurements (current and previous)
- Time interval ≥3 months for reliability
- Use maximum diameter (not volume) for consistency

```python
def compute_growth_velocity_comprehensive(visits: List[Visit]) -> None:
    """
    Compute growth velocity for all visits in a patient's timeline.
    
    Algorithm:
        - First visit: assign 'stable' (no prior measurement)
        - Subsequent visits: compute from last 2 measurements
        - Post-surgery: reset baseline, cannot compute until 2nd post-op measurement
    """
    for i, visit in enumerate(visits):
        if i == 0:
            # First visit: no prior measurement
            visit.growth_velocity_bucket = 'stable'
            
        elif visit.is_first_postsurgery_visit:
            # Surgery occurred: reset baseline
            visit.growth_velocity_bucket = 'stable'
            
        else:
            # Find last valid measurement
            prev_visit = visits[i - 1]
            
            # Check if both measurements available
            if visit.tumor_size_cm is not None and prev_visit.tumor_size_cm is not None:
                time_interval = visit.months_since_diagnosis - prev_visit.months_since_diagnosis
                
                if time_interval >= 3.0:  # Require ≥3 months
                    visit.growth_velocity_bucket = compute_growth_velocity_bucket(
                        current_size_cm=visit.tumor_size_cm,
                        previous_size_cm=prev_visit.tumor_size_cm,
                        time_interval_months=time_interval
                    )
                else:
                    # Interval too short: carry forward
                    visit.growth_velocity_bucket = prev_visit.growth_velocity_bucket
            else:
                # Missing measurement: carry forward
                visit.growth_velocity_bucket = prev_visit.growth_velocity_bucket
```

**Post-surgery handling**:
- Surgery dramatically reduces tumor size
- Cannot compare pre-surgery and post-surgery sizes directly
- Reset growth computation: first post-op visit = 'stable', then compute normally

---

### 6.2 Treatment Phase Computation

```python
def compute_all_treatment_phases(visits: List[Visit]) -> None:
    """
    Compute treatment phase for all visits based on intervention dates.
    
    Algorithm:
        1. Find all intervention dates (surgery, radiation)
        2. For each visit, determine phase based on time since last intervention
        3. Check for recurrence detection
    """
    # Find intervention dates
    surgery_date = None
    radiation_date = None
    recurrence_date = None
    
    for visit in visits:
        if visit.surgery_performed and surgery_date is None:
            surgery_date = visit.months_since_diagnosis
        if visit.radiation_performed and radiation_date is None:
            radiation_date = visit.months_since_diagnosis
        if visit.recurrence_detected and recurrence_date is None:
            recurrence_date = visit.months_since_diagnosis
    
    # Compute phase for each visit
    for visit in visits:
        visit.treatment_phase = compute_treatment_phase(
            visit_date_months=visit.months_since_diagnosis,
            surgery_date_months=surgery_date,
            radiation_date_months=radiation_date,
            recurrence_detected=(recurrence_date is not None and visit.months_since_diagnosis >= recurrence_date)
        )
```

**Priority rules**:
1. Recurrence overrides all other phases
2. Surgery takes priority over radiation (if both occurred)
3. Most recent intervention determines phase

---

### 6.3 Time Since Diagnosis

All temporal measurements are **relative to diagnosis date** (months since diagnosis).

```python
def compute_months_since_diagnosis(visit_date: datetime, diagnosis_date: datetime) -> float:
    """
    Compute months since diagnosis.
    
    Returns:
        Float representing months (allows fractional months)
    """
    days_diff = (visit_date - diagnosis_date).days
    months_diff = days_diff / 30.44  # Average days per month
    return months_diff
```

**Why months, not days?**
- Clinical thinking: "6-month follow-up", not "182-day follow-up"
- Reduces precision noise
- Aligns with bucketed observation actions

---

## 7. DATA FLOW PIPELINE

### 7.1 Overview

```
Raw Clinical Notes → Extraction → Post-Processing → Graph Construction → Output
```

### 7.2 Stage 1: Extraction from Clinical Notes

**Input**: Longitudinal clinical notes for each patient

**Process**:
```python
class RawVisit:
    """Raw extracted data from clinical note (before post-processing)."""
    patient_id: str
    visit_date: datetime
    
    # Extracted fields (may be missing)
    tumor_size_cm: Optional[float]
    symptoms_text: Optional[str]
    surgery_performed: bool
    surgery_type: Optional[str]
    radiation_performed: bool
    radiation_type: Optional[str]
    grade_from_pathology: Optional[str]
    recurrence_noted: bool
    
    # Derived
    months_since_diagnosis: float
```

**Extraction methods**:
- LLM-based extraction (GPT-4, Claude)
- Regex patterns for structured fields
- NER (Named Entity Recognition) for medications, procedures

**Output**: List of `RawVisit` objects per patient

---

### 7.3 Stage 2: Post-Processing

**Purpose**: Fill missing data, compute derived variables, canonicalize values

#### 7.3.1 Imputation Rules

```python
def impute_missing_data(visits: List[RawVisit]) -> List[Visit]:
    """
    Impute missing tumor sizes and symptoms using clinical logic.
    """
    # Sort by date
    visits.sort(key=lambda v: v.months_since_diagnosis)
    
    for i, visit in enumerate(visits):
        # Impute tumor size
        if visit.tumor_size_cm is None:
            if i > 0 and not visit.surgery_performed:
                # Carry forward from previous visit
                visit.tumor_size_cm = visits[i-1].tumor_size_cm
            elif visit.surgery_performed:
                # Post-surgery: assume 50% reduction for GTR, 70% for STR
                if i > 0:
                    if visit.surgery_type == 'gtr':
                        visit.tumor_size_cm = visits[i-1].tumor_size_cm * 0.5
                    else:
                        visit.tumor_size_cm = visits[i-1].tumor_size_cm * 0.7
        
        # Impute symptoms
        if visit.symptoms_text is None:
            if i > 0:
                # Carry forward
                visit.symptoms_text = visits[i-1].symptoms_text
            else:
                # Default: assume asymptomatic if not mentioned
                visit.symptoms_text = "no symptoms"
        
        # Impute grade
        if visit.grade_from_pathology is None:
            if i > 0 and visits[i-1].grade_from_pathology is not None:
                # Carry forward confirmed grade
                visit.grade_from_pathology = visits[i-1].grade_from_pathology
            else:
                # Default: assume grade 1 until pathology available
                visit.grade_from_pathology = 'grade_1'
    
    return visits
```

#### 7.3.2 Bucket Assignment

```python
def assign_buckets(visits: List[Visit]) -> None:
    """
    Convert raw values to bucket assignments.
    """
    for visit in visits:
        visit.tumor_size_bucket = compute_tumor_size_bucket(visit.tumor_size_cm)
        visit.symptoms_bucket = compute_symptom_bucket(visit.symptoms_text)
        visit.grade_bucket = compute_grade_bucket(visit.grade_from_pathology)
```

#### 7.3.3 Derived Variable Computation

```python
def compute_derived_variables(visits: List[Visit]) -> None:
    """
    Compute growth velocity and treatment phase.
    """
    compute_growth_velocity_comprehensive(visits)
    compute_all_treatment_phases(visits)
```

**Output**: List of fully processed `Visit` objects

---

### 7.4 Stage 3: Graph Construction

```python
def construct_graphs(all_patients: List[PatientPathway]) -> Dict[StratificationKey, Graph]:
    """
    Build 90 separate MDP graphs from patient pathways.
    
    Returns:
        Dictionary mapping StratificationKey → Graph object
    """
    graphs = {}
    
    # Initialize 90 empty graphs
    for age in AGE_BUCKETS:
        for gender in GENDER_BUCKETS:
            for grade in GRADE_BUCKETS:
                for location in LOCATION_BUCKETS:
                    key = StratificationKey(age, gender, grade, location)
                    graphs[key] = Graph(stratification_key=key)
    
    # Build transitions
    all_transitions = build_transitions(all_patients)
    
    # Add transitions to appropriate graphs
    for transition in all_transitions:
        graph = graphs[transition.from_graph]
        graph.add_transition(transition)
        
        # If cross-graph, also note in destination graph
        if transition.is_cross_graph:
            to_graph = graphs[transition.to_graph]
            to_graph.add_incoming_cross_graph_transition(transition)
    
    return graphs
```

**Output**: 90 `Graph` objects

---

### 7.5 Stage 4: Output Generation

**Multiple output formats**:

1. **NetworkX graphs** (.gpickle)
2. **JSON** (human-readable, for visualization)
3. **CSV edge lists** (for analysis in R/Python)
4. **Interactive visualizations** (Plotly, D3.js)
5. **Summary statistics** (markdown report)

---

## 8. EDGE CASES & VALIDATION RULES

### 8.1 Missing Data Handling

#### 8.1.1 Missing Tumor Size

**Scenarios**:
1. **Imaging not done at visit**: Carry forward from previous visit
2. **First visit, no prior data**: Cannot proceed (require baseline measurement)
3. **Post-surgery, no measurement**: Impute 50% reduction (GTR) or 70% (STR)

```python
if visit.tumor_size_cm is None:
    if is_first_visit:
        raise ValueError("Cannot proceed without baseline tumor size")
    elif visit.surgery_performed:
        visit.tumor_size_cm = prev_visit.tumor_size_cm * (0.5 if gtr else 0.7)
    else:
        visit.tumor_size_cm = prev_visit.tumor_size_cm  # Carry forward
```

#### 8.1.2 Missing Symptoms

**Default assumption**: Asymptomatic if not mentioned

```python
if visit.symptoms_text is None or visit.symptoms_text == "":
    visit.symptoms_bucket = 'none'
```

#### 8.1.3 Missing Grade

**Before surgery**: Assume Grade 1 (most common)
**After surgery**: Use pathology-confirmed grade, never revert to assumption

```python
if visit.grade_from_pathology is None:
    if any_prior_pathology_available:
        visit.grade_bucket = last_confirmed_grade
    else:
        visit.grade_bucket = 'grade_1'  # Assumption
```

---

### 8.2 Contradictory Information

#### 8.2.1 Size Increase Post-Surgery

**Problem**: Post-surgery size > pre-surgery size (measurement error or residual growth)

**Resolution**:
```python
if post_surgery_size > pre_surgery_size:
    # Likely measurement error; cap at pre-surgery size
    post_surgery_size = pre_surgery_size * 0.9
    warnings.append("Size increase post-surgery; likely measurement error")
```

#### 8.2.2 Symptom Resolution Without Intervention

**Problem**: Symptoms disappear without surgery/radiation

**Resolution**: Allow (symptoms can fluctuate)
```python
# This is valid; symptoms may wax and wane
# Do not force symptom persistence
```

#### 8.2.3 Grade Downgrade

**Problem**: Grade 2 → Grade 1 on repeat pathology

**Resolution**: Use most recent pathology
```python
# Rare but possible (initial biopsy sampled atypical area)
# Trust most recent pathology
visit.grade_bucket = most_recent_pathology_grade
```

---

### 8.3 Grade Uncertainty Before Pathology

**Problem**: Grade unknown for treatment-naive patients without surgery

**Solution**: Create **shadow graph** with `grade_unknown`, then consolidate after pathology

**Alternative (simpler)**: Assume Grade 1 until pathology available
- Pro: Simpler implementation
- Con: May create false "grade upgrade" transitions
- **Recommendation**: Use assumption method; mark as "assumed grade" in metadata

---

### 8.4 Validation Checks

#### 8.4.1 Transition Validity

```python
def validate_transition(transition: Transition) -> List[str]:
    """
    Check if transition is clinically plausible.
    
    Returns list of warnings (empty if valid).
    """
    warnings = []
    
    # Check 1: Action available in from_phase?
    if not is_action_available(transition.action, transition.from_state.treatment_phase):
        warnings.append(f"Action {transition.action} not available in phase {transition.from_state.treatment_phase}")
    
    # Check 2: Treatment phase transition makes sense?
    if transition.action in ['surgery_gtr', 'surgery_str']:
        if transition.to_state.treatment_phase != 'early_postop':
            warnings.append(f"Surgery should transition to early_postop, got {transition.to_state.treatment_phase}")
    
    # Check 3: Size change plausible?
    if transition.from_state.tumor_size == 'large' and transition.to_state.tumor_size == 'small':
        if transition.action not in ['surgery_gtr', 'surgery_str']:
            warnings.append(f"Large→small transition without surgery: {transition.action}")
    
    # Check 4: Symptom resolution without intervention?
    if transition.from_state.symptoms == 'present' and transition.to_state.symptoms == 'none':
        if transition.action.startswith('observe'):
            warnings.append("Symptom resolution with observation only (unusual but possible)")
    
    return warnings
```

#### 8.4.2 Patient Timeline Consistency

```python
def validate_patient_timeline(visits: List[Visit]) -> List[str]:
    """
    Check patient timeline for inconsistencies.
    """
    warnings = []
    
    # Check 1: Monotonic time
    for i in range(len(visits) - 1):
        if visits[i+1].months_since_diagnosis <= visits[i].months_since_diagnosis:
            warnings.append(f"Non-monotonic time at visit {i+1}")
    
    # Check 2: Phase progression logical
    for i in range(len(visits) - 1):
        if visits[i].treatment_phase == 'naive' and visits[i+1].treatment_phase == 'late_postop':
            warnings.append(f"Skipped early_postop phase at visit {i+1}")
    
    # Check 3: Recurrence only after treatment
    for i, visit in enumerate(visits):
        if visit.treatment_phase == 'recurrent':
            prior_treatment = any(v.treatment_phase in ['early_postop', 'late_postop', 'early_postrad', 'late_postrad'] for v in visits[:i])
            if not prior_treatment:
                warnings.append(f"Recurrence without prior treatment at visit {i}")
    
    return warnings
```

---

## 9. OUTPUT FORMAT SPECIFICATION

### 9.1 Graph Object Structure

```python
class Graph:
    """
    MDP graph for a single stratification.
    """
    stratification_key: StratificationKey
    nodes: Dict[StateID, Node]
    edges: List[Transition]
    
    # Metadata
    patient_count: int
    total_observations: int
    coverage: float  # Fraction of 108 states with ≥1 observation
    
    def add_node(self, state_id: StateID):
        """Add a state node to the graph."""
        pass
    
    def add_transition(self, transition: Transition):
        """Add a transition edge to the graph."""
        pass
    
    def get_available_actions(self, state_id: StateID) -> List[str]:
        """Get actions available from state."""
        pass
    
    def get_transition_probability(self, from_state: StateID, action: str, to_state: StateID) -> float:
        """Compute P(to_state | from_state, action)."""
        pass
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph."""
        pass
    
    def to_json(self) -> Dict:
        """Convert to JSON for visualization."""
        pass
```

### 9.2 JSON Output Format

```json
{
  "stratification": {
    "age_bucket": "<50",
    "gender": "F",
    "grade": "grade_1",
    "location": "convexity"
  },
  "metadata": {
    "patient_count": 12,
    "total_observations": 187,
    "state_coverage": 0.43,
    "edge_coverage": 0.18
  },
  "nodes": [
    {
      "state_id": "small_none_stable_naive",
      "tumor_size": "small",
      "symptoms": "none",
      "growth_velocity": "stable",
      "treatment_phase": "naive",
      "observation_count": 45,
      "patient_ids": ["P001", "P003", "P007"]
    }
  ],
  "edges": [
    {
      "from_state": "small_none_stable_naive",
      "action": "observe_medium",
      "to_state": "small_none_stable_naive",
      "count": 38,
      "mean_time_elapsed_months": 6.2,
      "std_time_elapsed_months": 1.1,
      "outcomes": {
        "stable": 36,
        "progression": 2
      }
    }
  ],
  "cross_graph_transitions": [
    {
      "from_graph": "<50_F_grade_1_convexity",
      "from_state": "medium_present_slow_growth_late_postop",
      "action": "observe_medium",
      "to_graph": "<50_F_grade_2_convexity",
      "to_state": "medium_present_fast_growth_recurrent",
      "count": 1,
      "changed_factors": ["tumor_grade"]
    }
  ]
}
```

### 9.3 Summary Statistics Output

```markdown
# Knowledge Graph Summary

## Stratification: <50_F_grade_1_convexity

### Data Coverage
- Patients: 12
- Total observations: 187
- States observed: 47 / 108 (43.5%)
- Edges observed: 94 / 864 (10.9%)

### Most Common States
1. small_none_stable_naive: 45 observations (24.1%)
2. small_none_stable_late_postop: 32 observations (17.1%)
3. medium_present_slow_growth_naive: 18 observations (9.6%)

### Most Common Transitions
1. small_none_stable_naive --[observe_medium]--> small_none_stable_naive: 38 times
2. small_none_stable_late_postop --[observe_long]--> small_none_stable_late_postop: 28 times
3. medium_present_slow_growth_naive --[surgery_gtr]--> small_none_stable_early_postop: 12 times

### Treatment Patterns
- Surgery rate: 45.8% (86/187 observations resulted in surgery)
- Radiation rate: 8.0% (15/187)
- Observation: 81.3% (152/187)

### Guideline Deviations Detected
- 3 instances of observe_long in early_postop phase (should be observe_short)
- 1 instance of symptom resolution without intervention
```

---

## 10. IMPLEMENTATION NOTES

### 10.1 Recommended Tech Stack

- **Python 3.9+**
- **Core**: pandas, numpy
- **Graphs**: networkx
- **Visualization**: plotly, matplotlib
- **NLP**: openai API (GPT-4) or anthropic API (Claude)
- **Data validation**: pydantic
- **Testing**: pytest

### 10.2 Processing Order

1. Extract raw visits from clinical notes (LLM + regex)
2. Compute derived fields (months since diagnosis, etc.)
3. Impute missing data
4. Assign buckets
5. Compute growth velocity
6. Compute treatment phase
7. Construct state IDs
8. Build transitions
9. Aggregate into graphs
10. Validate
11. Generate outputs

### 10.3 Performance Considerations

- **Batch processing**: Process patients in parallel
- **Caching**: Cache LLM extractions to avoid re-calls
- **Memory**: 50 patients × 35 visits = 1,750 visits fits easily in memory
- **Graph storage**: Use sparse matrix representation for transition probabilities

### 10.4 Testing Strategy

1. **Unit tests**: Individual bucket computation functions
2. **Integration tests**: Full pipeline on synthetic data
3. **Validation tests**: Clinical plausibility checks
4. **Regression tests**: Compare outputs before/after code changes

---

## 11. CLINICAL INTERPRETATION GUIDE

### 11.1 Reading State Representations

**Example state**: `medium_present_slow_growth_early_postop`

**Interpretation**:
- **medium**: Tumor is 3-5cm (surgical intervention was likely appropriate)
- **present**: Patient has neurological symptoms
- **slow_growth**: Growing 2-5mm/year (standard meningioma growth)
- **early_postop**: 0-6 months after surgery (high-risk period for complications)

**Clinical context**: This patient recently underwent surgery for a symptomatic, moderately-sized, growing tumor. They are in early post-operative surveillance phase and should be monitored closely (observe_short or observe_medium).

### 11.2 Reading Transitions

**Example transition**: 
```
large_present_growing_naive --[surgery_gtr]--> small_none_stable_early_postop
Count: 15
Mean time elapsed: 3.2 months
```

**Interpretation**:
- **From state**: Large, symptomatic, fast-growing tumor in treatment-naive patient
- **Action**: Gross total resection performed
- **To state**: Small residual, asymptomatic, stable tumor in early post-op phase
- **Frequency**: Observed 15 times in dataset
- **Outcome**: Successful surgery (symptom resolution, size reduction)

**Clinical context**: This is the expected, ideal outcome for surgical treatment of large symptomatic meningiomas.

### 11.3 Identifying Practice Patterns

Query: "What surveillance intervals are used in early post-op phase?"

```python
early_postop_actions = graph.filter(treatment_phase='early_postop').actions
print(Counter(early_postop_actions))
# {'observe_short': 78%, 'observe_medium': 20%, 'observe_long': 2%}
```

**Interpretation**: Most clinicians (78%) use aggressive 3-month surveillance in early post-op, consistent with NCCN guidelines. However, 2% use yearly surveillance, which may represent guideline deviation.

---

## 12. FUTURE EXTENSIONS

### 12.1 Q-Value Computation

Once graphs are built, standard MDP algorithms can compute optimal policies:

```python
def value_iteration(graph: Graph, gamma: float = 0.95, theta: float = 0.01) -> Dict[StateID, float]:
    """
    Compute state values via value iteration.
    
    V(s) = max_a Σ_s' P(s'|s,a) [R(s,a,s') + γ V(s')]
    """
    # Define reward function (e.g., 1 for good outcome, 0 for bad)
    # Run value iteration
    # Return V(s) for all states
    pass
```

### 12.2 Counterfactual Analysis

```python
def compare_outcomes(graph: Graph, state: StateID, action1: str, action2: str) -> Dict:
    """
    Compare outcomes of two actions from same state.
    
    Returns:
        {
            'action1': {'mean_recurrence_rate': 0.08, ...},
            'action2': {'mean_recurrence_rate': 0.15, ...}
        }
    """
    pass
```

### 12.3 Guideline Deviation Detection

```python
def detect_guideline_deviations(graph: Graph, guidelines: Dict) -> List[Deviation]:
    """
    Compare observed actions to NCCN guidelines.
    
    Example:
        NCCN: Grade 2+ requires post-op surveillance ≤6 months
        Observed: observe_long in early_postop phase
        → FLAG as deviation
    """
    pass
```

---

## APPENDIX A: QUICK REFERENCE

### Bucket Definitions Summary

| Variable | Buckets | Thresholds | Source |
|----------|---------|------------|--------|
| Age | 3: <50, 50-65, ≥65 | 50, 65 years | Diagnosis date |
| Gender | 2: M, F | - | Demographics |
| Grade | 3: grade_1, grade_2, grade_3 | WHO classification | Pathology |
| Location | 5: convexity, skull_base, parasagittal, sphenoid_wing, other | Anatomical | MRI |
| Tumor Size | 3: small, medium, large | 3cm, 5cm | MRI |
| Symptoms | 2: none, present | - | Clinical note |
| Growth Velocity | 3: stable, slow, fast | 2mm/yr, 5mm/yr | Sequential MRIs |
| Treatment Phase | 6: naive, early_postop, late_postop, early_postrad, late_postrad, recurrent | 6 months | Intervention dates |

### Action Definitions

1. `observe_short`: 0-4.5 months
2. `observe_medium`: 4.5-9 months
3. `observe_long`: 9+ months
4. `surgery_gtr`: Gross total resection
5. `surgery_str`: Subtotal resection
6. `radiation_srs`: Stereotactic radiosurgery
7. `radiation_fsrt`: Fractionated radiotherapy
8. `supportive_care`: Symptom management only

### State Space Size

- Stratification graphs: 90
- States per graph: 3 × 2 × 3 × 6 = 108
- Total states: 9,720
- Actions: 8
- Expected edges: ~17,280 (with phase constraints)

---

## APPENDIX B: GLOSSARY

| Term | Definition |
|------|------------|
| **MDP** | Markov Decision Process: (S, A, T, R) |
| **Stratification** | Grouping patients by static demographics |
| **State** | Current patient condition (dynamic variables) |
| **Action** | Clinical decision at visit |
| **Transition** | Evolution from one state to another |
| **Cross-graph transition** | Grade change causing move between graphs |
| **GTR** | Gross Total Resection (Simpson I-II) |
| **STR** | Subtotal Resection (Simpson III-IV) |
| **SRS** | Stereotactic Radiosurgery (single-session) |
| **FSRT** | Fractionated Stereotactic Radiotherapy |
| **WHO Grade** | World Health Organization tumor grade (1-3) |
| **Convexity** | Surface of brain under skull |
| **Skull base** | Bottom of skull near cranial nerves |
| **NCCN** | National Comprehensive Cancer Network (guidelines) |

---

**END OF SPECIFICATION**

Version: 2.0  
Last Updated: February 10, 2026  
Contact: [Your contact info]
