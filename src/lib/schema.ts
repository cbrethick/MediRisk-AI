/**
 * The questionnaire.
 *
 * Every question here maps to a real MEPS variable that the model was trained
 * on, and every question carries a `why` string that is shown in the interface.
 * The rule for this project is that the person answering should never be asked
 * for something without being told what it is used for.
 */
import {
  Activity, Baby, Bike, Brain, Briefcase, CigaretteOff, Droplets, Gauge,
  GraduationCap, HeartPulse, Home, Landmark, MapPin, Ribbon, Scale, ShieldCheck,
  Stethoscope, User, Users, Wind, Zap,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

export interface Choice {
  value: number;
  label: string;
  hint?: string;
  icon?: LucideIcon;
}

export interface Question {
  key: string;
  label: string;
  why: string;                       // shown under the question, always
  icon: LucideIcon;
  kind: "choice" | "number" | "slider";
  choices?: Choice[];
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  optional?: boolean;                // may be left as "prefer not to say"
}

export interface Step {
  id: string;
  title: string;
  blurb: string;
  icon: LucideIcon;
  questions: Question[];
}

export const CONDITION_KEYS = [
  "HIBPDX", "DIABDX_M18", "CHDDX", "CANCERDX",
  "ASTHDX", "ARTHDX", "CHOLDX", "STRKDX",
];

export const CONDITIONS: { key: string; label: string; icon: LucideIcon; note: string }[] = [
  { key: "HIBPDX", label: "High blood pressure", icon: Gauge, note: "Ever told by a doctor" },
  { key: "CHOLDX", label: "High cholesterol", icon: Droplets, note: "Ever told by a doctor" },
  { key: "ARTHDX", label: "Arthritis", icon: Activity, note: "Any type, ever diagnosed" },
  { key: "DIABDX_M18", label: "Diabetes", icon: Zap, note: "Type 1 or type 2" },
  { key: "ASTHDX", label: "Asthma", icon: Wind, note: "Ever diagnosed" },
  { key: "CHDDX", label: "Coronary heart disease", icon: HeartPulse, note: "Including angina" },
  { key: "CANCERDX", label: "Cancer", icon: Ribbon, note: "Any cancer, ever diagnosed" },
  { key: "STRKDX", label: "Stroke", icon: Brain, note: "Ever had a stroke" },
];

export const STEPS: Step[] = [
  {
    id: "about",
    title: "About you",
    blurb: "Three basics. Age is one of the strongest single predictors of healthcare cost.",
    icon: User,
    questions: [
      {
        key: "age", label: "How old are you?", icon: User, kind: "slider",
        min: 18, max: 85, step: 1, unit: "years",
        why: "Spending rises steeply with age. The model was trained on adults only, because the survey collects BMI and smoking for adults, so 18 is the floor and MEPS tops its age variable out at 85.",
      },
      {
        key: "sex", label: "Sex recorded on your health record", icon: Users, kind: "choice",
        choices: [
          { value: 1, label: "Male" },
          { value: 2, label: "Female" },
        ],
        why: "Used as a model input and also audited afterwards: the Research page reports error rates separately for men and women so you can see whether the model treats them differently.",
      },
      {
        key: "region", label: "Which region do you live in?", icon: MapPin, kind: "choice",
        choices: [
          { value: 1, label: "Northeast" },
          { value: 2, label: "Midwest" },
          { value: 3, label: "South" },
          { value: 4, label: "West" },
        ],
        why: "US healthcare prices vary a lot by region, so the same treatment produces a different bill depending on where it happens. These are the four US Census regions MEPS uses.",
      },
    ],
  },
  {
    id: "body",
    title: "Body and lifestyle",
    blurb: "Height and weight give BMI. Smoking combined with a high BMI is the interaction this project was built to capture.",
    icon: Scale,
    questions: [
      {
        key: "bmi", label: "Body mass index", icon: Scale, kind: "number",
        min: 12, max: 60, step: 0.1, optional: true,
        why: "Enter your height and weight and this is calculated for you. BMI is missing for about 30% of adults in the survey, so the model has an explicit way of handling 'not recorded' and you can skip it.",
      },
      {
        key: "smoking_status", label: "Do you smoke?", icon: CigaretteOff, kind: "choice",
        choices: [
          { value: 1, label: "Yes, every day" },
          { value: 2, label: "Yes, some days" },
          { value: 3, label: "No, not at all" },
        ],
        why: "Smoking on its own raises expected cost, and the model also carries a separate smoker-and-obese term, because the two together cost more than adding their individual effects would suggest.",
      },
      {
        key: "exercise_days", label: "Days a week with 30+ minutes of moderate activity", icon: Bike,
        kind: "slider", min: 0, max: 7, step: 1, unit: "days", optional: true,
        why: "A protective factor. The survey asks this only of adults who completed the self-administered questionnaire, so roughly 29% of people have no answer here.",
      },
    ],
  },
  {
    id: "health",
    title: "Your health",
    blurb: "Tick anything a doctor has ever told you that you have. Leave the rest untouched.",
    icon: Stethoscope,
    questions: [
      {
        key: "self_rated_health", label: "In general, how is your health?", icon: HeartPulse,
        kind: "choice",
        choices: [
          { value: 1, label: "Excellent" },
          { value: 2, label: "Very good" },
          { value: 3, label: "Good" },
          { value: 4, label: "Fair" },
          { value: 5, label: "Poor" },
        ],
        why: "One of the single most informative questions in the whole form. How people rate their own health predicts their spending better than most individual diagnoses do.",
      },
    ],
  },
  {
    id: "context",
    title: "Your circumstances",
    blurb: "Coverage and income shape how much care a person actually uses, separately from how sick they are.",
    icon: Landmark,
    questions: [
      {
        key: "insurance", label: "What health coverage do you have?", icon: ShieldCheck,
        kind: "choice",
        choices: [
          { value: 1, label: "Private insurance", hint: "Any private plan during the year" },
          { value: 2, label: "Public only", hint: "Medicare, Medicaid or similar" },
          { value: 3, label: "Uninsured", hint: "No coverage during the year" },
        ],
        why: "Coverage changes recorded spending in two directions at once: insured people use more care, but uninsured people who do get care are often sicker when they arrive.",
      },
      {
        key: "poverty_category", label: "Where does your family income sit?", icon: Briefcase,
        kind: "choice",
        choices: [
          { value: 1, label: "Poor or negative", hint: "Below the federal poverty line" },
          { value: 2, label: "Near poor", hint: "100–125% of the line" },
          { value: 3, label: "Low income", hint: "125–200%" },
          { value: 4, label: "Middle income", hint: "200–400%" },
          { value: 5, label: "High income", hint: "400% and above" },
        ],
        why: "Expressed as a multiple of the US federal poverty line, which is how MEPS records it. The Research page reports the model's error rate for each of these bands.",
      },
      {
        key: "marital_status", label: "Marital status", icon: Home, kind: "choice",
        choices: [
          { value: 1, label: "Married" },
          { value: 2, label: "Widowed" },
          { value: 3, label: "Divorced" },
          { value: 4, label: "Separated" },
          { value: 5, label: "Never married" },
          { value: 6, label: "Under 16 / not applicable", hint: "Rare in an adult sample" },
        ],
        why: "A modest but real predictor, mostly because it tracks household support and whether someone has coverage through a partner.",
      },
      {
        key: "education_years", label: "Years of education completed", icon: GraduationCap,
        kind: "slider", min: 0, max: 17, step: 1, unit: "years", optional: true,
        why: "12 is a high school diploma, 16 a bachelor's degree, 17 anything beyond. It acts as a proxy for health literacy and for how readily someone seeks care.",
      },
    ],
  },
];

/**
 * Raw answers for a brand-new session.
 *
 * Everything starts unanswered EXCEPT age, which is pre-set to the median adult
 * age in the survey. A slider that reads "—" while its handle sits in the
 * middle is a small lie about the current state, and age is the one question
 * with no "prefer not to say" option, so it starts at a real, labelled value.
 */
export function emptyAnswers(): Record<string, number | null> {
  const a: Record<string, number | null> = {};
  for (const s of STEPS) for (const q of s.questions) a[q.key] = null;
  for (const k of CONDITION_KEYS) a[k] = null;
  a.age = 45;
  return a;
}

export const FEATURE_LABELS: Record<string, string> = {
  age: "Age",
  sex: "Sex",
  bmi: "Body mass index",
  region: "Region",
  smoking_status: "Smoking",
  exercise_days: "Physical activity",
  self_rated_health: "Self-rated health",
  insurance: "Insurance coverage",
  poverty_category: "Income band",
  marital_status: "Marital status",
  education_years: "Education",
  chronic_count: "Number of chronic conditions",
  is_obese: "Obesity (BMI ≥ 30)",
  is_smoker: "Current smoker",
  smoker_and_obese: "Smoking combined with obesity",
  is_senior: "Aged 65 or over",
  multimorbid: "Three or more conditions",
  HIBPDX: "High blood pressure",
  DIABDX_M18: "Diabetes",
  CHDDX: "Coronary heart disease",
  CANCERDX: "Cancer",
  ASTHDX: "Asthma",
  ARTHDX: "Arthritis",
  CHOLDX: "High cholesterol",
  STRKDX: "Stroke",
};
