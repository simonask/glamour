#[cfg(feature = "serde")]
mod serialization {
    use glamour::*;

    #[test]
    fn vector2() {
        assert_eq!(
            serde_json::to_string(&Vector2::<f32> { x: 1.0, y: 2.0 }).unwrap(),
            r#"{"x":1.0,"y":2.0}"#
        );
        assert_eq!(
            serde_json::from_str::<Vector2<f32>>(r#"{"x":1.0,"y":2.0}"#).unwrap(),
            Vector2::<f32> { x: 1.0, y: 2.0 }
        );
        assert_eq!(
            serde_json::from_str::<Vector2<f32>>(r#"[1.0,2.0]"#).unwrap(),
            Vector2::<f32> { x: 1.0, y: 2.0 }
        );
    }

    #[test]
    fn vector3() {
        assert_eq!(
            serde_json::to_string(&Vector3::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0
            })
            .unwrap(),
            r#"{"x":1.0,"y":2.0,"z":3.0}"#
        );
        assert_eq!(
            serde_json::from_str::<Vector3<f32>>(r#"{"x":1.0,"y":2.0,"z":3.0}"#).unwrap(),
            Vector3::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0
            }
        );
        assert_eq!(
            serde_json::from_str::<Vector3<f32>>(r#"[1.0,2.0,3.0]"#).unwrap(),
            Vector3::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0
            }
        );
    }

    #[test]
    fn vector4() {
        assert_eq!(
            serde_json::to_string(&Vector4::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                w: 4.0
            })
            .unwrap(),
            r#"{"x":1.0,"y":2.0,"z":3.0,"w":4.0}"#
        );
        assert_eq!(
            serde_json::from_str::<Vector4<f32>>(r#"{"x":1.0,"y":2.0,"z":3.0,"w":4.0}"#).unwrap(),
            Vector4::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                w: 4.0,
            }
        );
        assert_eq!(
            serde_json::from_str::<Vector4<f32>>(r#"[1.0,2.0,3.0,4.0]"#).unwrap(),
            Vector4::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                w: 4.0
            }
        );
    }

    #[test]
    fn point2() {
        assert_eq!(
            serde_json::to_string(&Point2::<f32> { x: 1.0, y: 2.0 }).unwrap(),
            r#"{"x":1.0,"y":2.0}"#
        );
        assert_eq!(
            serde_json::from_str::<Point2<f32>>(r#"{"x":1.0,"y":2.0}"#).unwrap(),
            Point2::<f32> { x: 1.0, y: 2.0 }
        );
        assert_eq!(
            serde_json::from_str::<Point2<f32>>(r#"[1.0,2.0]"#).unwrap(),
            Point2::<f32> { x: 1.0, y: 2.0 }
        );
    }

    #[test]
    fn point3() {
        assert_eq!(
            serde_json::to_string(&Point3::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0
            })
            .unwrap(),
            r#"{"x":1.0,"y":2.0,"z":3.0}"#
        );
        assert_eq!(
            serde_json::from_str::<Point3<f32>>(r#"{"x":1.0,"y":2.0,"z":3.0}"#).unwrap(),
            Point3::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0
            }
        );
        assert_eq!(
            serde_json::from_str::<Point3<f32>>(r#"[1.0,2.0,3.0]"#).unwrap(),
            Point3::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0
            }
        );
    }

    #[test]
    fn point4() {
        assert_eq!(
            serde_json::to_string(&Point4::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                w: 4.0
            })
            .unwrap(),
            r#"{"x":1.0,"y":2.0,"z":3.0,"w":4.0}"#
        );
        assert_eq!(
            serde_json::from_str::<Point4<f32>>(r#"{"x":1.0,"y":2.0,"z":3.0,"w":4.0}"#).unwrap(),
            Point4::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                w: 4.0,
            }
        );
        assert_eq!(
            serde_json::from_str::<Point4<f32>>(r#"[1.0,2.0,3.0,4.0]"#).unwrap(),
            Point4::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                w: 4.0
            }
        );
    }

    #[test]
    fn size2() {
        assert_eq!(
            serde_json::to_string(&Size2::<f32> {
                width: 1.0,
                height: 2.0
            })
            .unwrap(),
            r#"{"width":1.0,"height":2.0}"#
        );
        assert_eq!(
            serde_json::from_str::<Size2<f32>>(r#"{"width":1.0,"height":2.0}"#).unwrap(),
            Size2::<f32> {
                width: 1.0,
                height: 2.0
            }
        );
        assert_eq!(
            serde_json::from_str::<Size2<f32>>(r#"[1.0,2.0]"#).unwrap(),
            Size2::<f32> {
                width: 1.0,
                height: 2.0
            }
        );
    }

    #[test]
    fn size3() {
        assert_eq!(
            serde_json::to_string(&Size3::<f32> {
                width: 1.0,
                height: 2.0,
                depth: 3.0
            })
            .unwrap(),
            r#"{"width":1.0,"height":2.0,"depth":3.0}"#
        );
        assert_eq!(
            serde_json::from_str::<Size3<f32>>(r#"{"width":1.0,"height":2.0,"depth":3.0}"#)
                .unwrap(),
            Size3::<f32> {
                width: 1.0,
                height: 2.0,
                depth: 3.0
            }
        );
        assert_eq!(
            serde_json::from_str::<Size3<f32>>(r#"[1.0,2.0,3.0]"#).unwrap(),
            Size3::<f32> {
                width: 1.0,
                height: 2.0,
                depth: 3.0
            }
        );
    }
}
